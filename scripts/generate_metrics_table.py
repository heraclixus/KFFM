#!/usr/bin/env python3
"""
Generate a LaTeX table comparing NFE, velocity uniformity, and convergence rate
across datasets and kernel types.

Usage:
    python generate_metrics_table.py
    python generate_metrics_table.py --output ../outputs/metrics_table.tex
"""

import sys
sys.path.append('../')

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Configuration
# =============================================================================

# Kernels to include (in order)
KERNELS = ['none', 'rbf', 'euclidean']

# k-FFM kernels (excluding independent)
KFFM_KERNELS = ['rbf', 'euclidean', 'signature']

# Kernel display names
KERNEL_DISPLAY_NAMES = {
    'none': 'Independent',
    'rbf': 'RBF',
    'euclidean': 'Euclidean',
    'signature': 'Signature',
}

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

# Datasets to include
DEFAULT_DATASETS = [
    'aemet', 'rbergomi', 'heston', 'kdv', 
    'stochastic_kdv', 'navier_stokes', 'stochastic_ns', 'expr_genes', 'economy'
]


@dataclass
class MetricValue:
    """Container for a metric value with optional std."""
    mean: float
    std: float = 0.0
    
    def __str__(self) -> str:
        if self.std > 0:
            return f"{self.mean:.1f}±{self.std:.1f}"
        return f"{self.mean:.2f}"


# =============================================================================
# Data Loading
# =============================================================================

def load_velocity_analysis(analysis_dir: Path, dataset: str) -> Optional[Dict]:
    """Load velocity analysis JSON for a dataset."""
    json_path = analysis_dir / dataset / f'{dataset}_velocity_analysis.json'
    
    if not json_path.exists():
        return None
    
    with open(json_path) as f:
        return json.load(f)


def load_convergence_rate(seeded_dir: Path, dataset: str, kernel: str) -> Optional[Tuple[float, float]]:
    """
    Load convergence rate from quality_metrics.json files.
    Returns (mean, std) across all seeds.
    
    Handles two directory structures:
    - Standard: seeded_dir/dataset/kernel/seed_N/quality_metrics.json
    - Nested:   seeded_dir/dataset/kernel/{subdir}/seed_N/quality_metrics.json (e.g., economy)
    """
    kernel_dir = seeded_dir / dataset / kernel
    
    if not kernel_dir.exists():
        return None
    
    rates = []
    
    def collect_from_dir(search_dir: Path):
        """Collect convergence rates from seed directories in search_dir."""
        for item in search_dir.iterdir():
            if not item.is_dir():
                continue
            
            if item.name.startswith('seed_'):
                # Found a seed directory
                metrics_path = item / 'quality_metrics.json'
                if metrics_path.exists():
                    try:
                        with open(metrics_path) as f:
                            metrics = json.load(f)
                            if 'convergence_rate' in metrics:
                                rates.append(metrics['convergence_rate'])
                    except:
                        pass
            else:
                # Check if this is a subdirectory containing seed_* dirs (nested structure)
                # e.g., economy/rbf/econ1_population/seed_1/
                for nested_item in item.iterdir():
                    if nested_item.is_dir() and nested_item.name.startswith('seed_'):
                        metrics_path = nested_item / 'quality_metrics.json'
                        if metrics_path.exists():
                            try:
                                with open(metrics_path) as f:
                                    metrics = json.load(f)
                                    if 'convergence_rate' in metrics:
                                        rates.append(metrics['convergence_rate'])
                            except:
                                pass
    
    collect_from_dir(kernel_dir)
    
    if not rates:
        return None
    
    return np.mean(rates), np.std(rates)


def extract_all_metrics(
    velocity_dir: Path,
    seeded_dir: Path,
    datasets: List[str],
    kernels: List[str],
) -> Dict[str, Dict[str, Dict[str, MetricValue]]]:
    """
    Extract all metrics for all dataset/kernel combinations.
    
    Returns:
        Dict[dataset][kernel][metric] = MetricValue
        where metric is one of: 'nfe', 'uniformity', 'convergence_rate'
    """
    data = {}
    
    for dataset in datasets:
        data[dataset] = {}
        
        # Load velocity analysis
        velocity_data = load_velocity_analysis(velocity_dir, dataset)
        
        for kernel in kernels:
            data[dataset][kernel] = {}
            
            # NFE and Uniformity from velocity analysis
            if velocity_data:
                result = velocity_data.get(kernel)
                if result:
                    if 'aggregated' in result:
                        data[dataset][kernel]['nfe'] = MetricValue(
                            result['aggregated']['nfe_mean'],
                            result['aggregated']['nfe_std']
                        )
                        data[dataset][kernel]['uniformity'] = MetricValue(
                            result['aggregated']['uniformity_mean'],
                            result['aggregated']['uniformity_std']
                        )
                    else:
                        data[dataset][kernel]['nfe'] = MetricValue(
                            result['trajectory'].get('mean_nfe', 0)
                        )
                        data[dataset][kernel]['uniformity'] = MetricValue(
                            result['trajectory']['time_variance']
                        )
            
            # Convergence rate from quality metrics
            conv_result = load_convergence_rate(seeded_dir, dataset, kernel)
            if conv_result:
                data[dataset][kernel]['convergence_rate'] = MetricValue(
                    conv_result[0], conv_result[1]
                )
    
    return data


# =============================================================================
# Table Generation
# =============================================================================

def format_value(value: MetricValue, is_best: bool, metric: str, is_kffm: bool = False) -> str:
    """Format a metric value (mean only), with bold green if best, ForestGreen if k-FFM."""
    if metric == 'nfe':
        text = f"{value.mean:.1f}"
    elif metric == 'uniformity':
        # Use regular decimal notation
        if value.mean >= 1000:
            text = f"{value.mean:.0f}"
        elif value.mean >= 100:
            text = f"{value.mean:.1f}"
        elif value.mean >= 10:
            text = f"{value.mean:.2f}"
        else:
            text = f"{value.mean:.2f}"
    else:  # convergence_rate
        text = f"{value.mean:.4f}"
    
    # Apply k-FFM color first (ForestGreen)
    if is_kffm:
        text = f"\\textcolor{{ForestGreen}}{{{text}}}"
    
    # Then apply bold if best
    if is_best:
        return f"\\textbf{{{text}}}"
    return text


def find_best_kernel(data: Dict[str, Dict[str, MetricValue]], metric: str, lower_is_better: bool = True) -> str:
    """Find the kernel with the best value for a metric."""
    best_kernel = None
    best_value = float('inf') if lower_is_better else float('-inf')
    
    for kernel, metrics in data.items():
        if metric not in metrics:
            continue
        value = metrics[metric].mean
        if lower_is_better:
            if value < best_value:
                best_value = value
                best_kernel = kernel
        else:
            if value > best_value:
                best_value = value
                best_kernel = kernel
    
    return best_kernel


def find_best_kffm_value(
    kernel_data: Dict[str, Dict[str, MetricValue]], 
    metric: str, 
    kffm_kernels: List[str],
    lower_is_better: bool = True
) -> Tuple[Optional[str], Optional[MetricValue]]:
    """Find the best k-FFM kernel and its value for a metric."""
    best_kernel = None
    best_value = None
    best_mean = float('inf') if lower_is_better else float('-inf')
    
    for kernel in kffm_kernels:
        if kernel not in kernel_data or metric not in kernel_data[kernel]:
            continue
        value = kernel_data[kernel][metric]
        if lower_is_better:
            if value.mean < best_mean:
                best_mean = value.mean
                best_kernel = kernel
                best_value = value
        else:
            if value.mean > best_mean:
                best_mean = value.mean
                best_kernel = kernel
                best_value = value
    
    return best_kernel, best_value


def generate_main_table(
    data: Dict[str, Dict[str, Dict[str, MetricValue]]],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate main paper table: FFM vs best k-FFM.
    
    Format:
    Dataset | NFE (FFM/k-FFM) | Uniformity (FFM/k-FFM) | Conv. Rate (FFM/k-FFM)
    """
    col_spec = 'lccc'
    
    lines = [
        "% Requires: \\usepackage{booktabs} \\usepackage[table,dvipsnames]{xcolor}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comparison of FFM (Independent) vs k-FFM (best kernel) across datasets. "
        "Each cell shows FFM/\\textcolor{ForestGreen}{k-FFM} values. Best values are \\textbf{bold}. "
        "Lower is better for NFE and Uniformity; higher is better for Conv. Rate.}",
        "\\label{tab:ffm_vs_kffm}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        "Dataset & NFE & Uniformity & Conv. Rate \\\\",
        "\\midrule",
    ]
    
    # Data rows
    for dataset, kernel_data in data.items():
        row = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        
        # Get FFM (independent) values
        ffm_data = kernel_data.get('none', {})
        
        # For each metric, compare FFM vs best k-FFM
        for metric in ['nfe', 'uniformity', 'convergence_rate']:
            # Convergence rate: higher is better; NFE/Uniformity: lower is better
            lower_is_better = metric != 'convergence_rate'
            
            ffm_value = ffm_data.get(metric)
            _, best_kffm_value = find_best_kffm_value(kernel_data, metric, KFFM_KERNELS, lower_is_better=lower_is_better)
            
            if ffm_value and best_kffm_value:
                # Determine which is better
                if lower_is_better:
                    ffm_is_best = ffm_value.mean <= best_kffm_value.mean
                    kffm_is_best = best_kffm_value.mean <= ffm_value.mean
                else:  # higher is better (convergence_rate)
                    ffm_is_best = ffm_value.mean >= best_kffm_value.mean
                    kffm_is_best = best_kffm_value.mean >= ffm_value.mean
                
                # Handle ties
                if abs(ffm_value.mean - best_kffm_value.mean) < 1e-9:
                    ffm_is_best = True
                    kffm_is_best = True
                
                ffm_str = format_value(ffm_value, ffm_is_best, metric, is_kffm=False)
                kffm_str = format_value(best_kffm_value, kffm_is_best, metric, is_kffm=True)
                
                row += f" & {ffm_str}/{kffm_str}"
            elif ffm_value:
                row += f" & {format_value(ffm_value, True, metric, is_kffm=False)}/--"
            elif best_kffm_value:
                row += f" & --/{format_value(best_kffm_value, True, metric, is_kffm=True)}"
            else:
                row += " & --/--"
        
        row += " \\\\"
        lines.append(row)
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    table = '\n'.join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table)
        print(f"✓ Saved main table: {output_path}")
    
    return table


def generate_appendix_table(
    data: Dict[str, Dict[str, Dict[str, MetricValue]]],
    kernels: List[str],
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate full appendix table with all kernels.
    
    Format:
    Dataset | Independent | RBF | Euclidean | Signature
    where each cell contains: NFE/Uniformity/Conv.Rate
    """
    # Build table header
    n_kernels = len(kernels)
    col_spec = 'l' + 'c' * n_kernels
    
    # Header row with kernel names only
    header_row = "Dataset"
    for kernel in kernels:
        display_name = KERNEL_DISPLAY_NAMES.get(kernel, kernel)
        header_row += f" & {display_name}"
    
    lines = [
        "% Requires: \\usepackage{booktabs} \\usepackage[table,dvipsnames]{xcolor}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Full comparison of NFE / Velocity Uniformity / Convergence Rate across datasets and all kernel types. "
        "Best values are \\textbf{bold}. \\textcolor{ForestGreen}{Green} indicates k-FFM methods. "
        "Lower is better for NFE and Uniformity; higher is better for Conv. Rate.}",
        "\\label{tab:metrics_full}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        header_row + " \\\\",
        "\\midrule",
    ]
    
    # Data rows
    for dataset, kernel_data in data.items():
        # Find best for each metric across ALL kernels
        # NFE and Uniformity: lower is better; Convergence rate: higher is better
        best_nfe = find_best_kernel(kernel_data, 'nfe', lower_is_better=True)
        best_uniformity = find_best_kernel(kernel_data, 'uniformity', lower_is_better=True)
        best_convergence = find_best_kernel(kernel_data, 'convergence_rate', lower_is_better=False)
        
        row = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        
        for kernel in kernels:
            metrics = kernel_data.get(kernel, {})
            is_kffm = kernel != 'none'  # All kernels except 'none' are k-FFM
            
            # Build a/b/c format for this cell
            parts = []
            
            # NFE
            if 'nfe' in metrics:
                parts.append(format_value(metrics['nfe'], kernel == best_nfe, 'nfe', is_kffm=is_kffm))
            else:
                parts.append("--")
            
            # Uniformity
            if 'uniformity' in metrics:
                parts.append(format_value(metrics['uniformity'], kernel == best_uniformity, 'uniformity', is_kffm=is_kffm))
            else:
                parts.append("--")
            
            # Convergence rate
            if 'convergence_rate' in metrics:
                parts.append(format_value(metrics['convergence_rate'], kernel == best_convergence, 'convergence_rate', is_kffm=is_kffm))
            else:
                parts.append("--")
            
            row += " & " + "/".join(parts)
        
        row += " \\\\"
        lines.append(row)
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    table = '\n'.join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(table)
        print(f"✓ Saved appendix table: {output_path}")
    
    return table


def generate_markdown_table(
    data: Dict[str, Dict[str, Dict[str, MetricValue]]],
    kernels: List[str],
) -> str:
    """Generate a simple Markdown table for quick viewing."""
    lines = []
    
    # Header
    header = "| Dataset |"
    sep = "|:--------|"
    for kernel in kernels:
        display_name = KERNEL_DISPLAY_NAMES.get(kernel, kernel)
        header += f" {display_name} (NFE/Unif/Conv) |"
        sep += ":---:|"
    
    lines.append(header)
    lines.append(sep)
    
    # Data rows
    for dataset, kernel_data in data.items():
        best_nfe = find_best_kernel(kernel_data, 'nfe', lower_is_better=True)
        best_uniformity = find_best_kernel(kernel_data, 'uniformity', lower_is_better=True)
        best_convergence = find_best_kernel(kernel_data, 'convergence_rate', lower_is_better=True)
        
        row = f"| {DATASET_DISPLAY_NAMES.get(dataset, dataset)} |"
        
        for kernel in kernels:
            metrics = kernel_data.get(kernel, {})
            
            parts = []
            
            # NFE
            if 'nfe' in metrics:
                v = metrics['nfe']
                s = f"{v.mean:.1f}"
                if kernel == best_nfe:
                    s = f"**{s}**"
                parts.append(s)
            else:
                parts.append("--")
            
            # Uniformity
            if 'uniformity' in metrics:
                v = metrics['uniformity']
                if v.mean >= 1000:
                    s = f"{v.mean:.0f}"
                elif v.mean >= 100:
                    s = f"{v.mean:.1f}"
                else:
                    s = f"{v.mean:.2f}"
                if kernel == best_uniformity:
                    s = f"**{s}**"
                parts.append(s)
            else:
                parts.append("--")
            
            # Convergence rate
            if 'convergence_rate' in metrics:
                v = metrics['convergence_rate']
                s = f"{v.mean:.4f}"
                if kernel == best_convergence:
                    s = f"**{s}**"
                parts.append(s)
            else:
                parts.append("--")
            
            row += f" {'/'.join(parts)} |"
        
        lines.append(row)
    
    return '\n'.join(lines)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate metrics comparison table (NFE, uniformity, convergence rate)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=DEFAULT_DATASETS,
                       help='Datasets to include')
    parser.add_argument('--kernels', type=str, nargs='+',
                       default=['none', 'rbf', 'euclidean', 'signature'],
                       help='Kernels to include')
    parser.add_argument('--velocity-dir', type=str,
                       default='../outputs/velocity_analysis',
                       help='Directory containing velocity analysis results')
    parser.add_argument('--seeded-dir', type=str,
                       default='../outputs/seeded_runs',
                       help='Directory containing seeded run results')
    parser.add_argument('--output', type=str,
                       default='../outputs/metrics_table.tex',
                       help='Output path for LaTeX table')
    
    args = parser.parse_args()
    
    velocity_dir = Path(args.velocity_dir)
    seeded_dir = Path(args.seeded_dir)
    
    print("="*70)
    print("Metrics Table Generator")
    print("="*70)
    print(f"Velocity analysis dir: {velocity_dir}")
    print(f"Seeded runs dir: {seeded_dir}")
    print(f"Datasets: {args.datasets}")
    print(f"Kernels: {args.kernels}")
    print()
    
    # Extract all metrics
    print("Loading metrics...")
    data = extract_all_metrics(velocity_dir, seeded_dir, args.datasets, args.kernels)
    
    # Print summary
    print("\nData Summary:")
    print("-"*70)
    for dataset in args.datasets:
        print(f"\n{DATASET_DISPLAY_NAMES.get(dataset, dataset)}:")
        for kernel in args.kernels:
            metrics = data.get(dataset, {}).get(kernel, {})
            if metrics:
                parts = []
                if 'nfe' in metrics:
                    parts.append(f"NFE={metrics['nfe'].mean:.1f}")
                if 'uniformity' in metrics:
                    parts.append(f"Unif={metrics['uniformity'].mean:.2e}")
                if 'convergence_rate' in metrics:
                    parts.append(f"Conv={metrics['convergence_rate'].mean:.4f}")
                print(f"  {KERNEL_DISPLAY_NAMES.get(kernel, kernel)}: {', '.join(parts)}")
            else:
                print(f"  {KERNEL_DISPLAY_NAMES.get(kernel, kernel)}: No data")
    
    # Generate tables
    print("\n" + "="*70)
    print("Generated Tables")
    print("="*70)
    
    output_path = Path(args.output)
    
    # 1. Main table (FFM vs best k-FFM) - for paper
    main_table_path = output_path.parent / 'metrics_table_main.tex'
    main_table = generate_main_table(data, main_table_path)
    
    # 2. Appendix table (all kernels) - includes signature
    all_kernels = ['none', 'rbf', 'euclidean', 'signature']
    appendix_table_path = output_path.parent / 'metrics_table_appendix.tex'
    appendix_table = generate_appendix_table(data, all_kernels, appendix_table_path)
    
    # Markdown preview of main table
    print("\nMain Table Preview (FFM vs k-FFM):")
    print("-"*70)
    print("Dataset | NFE | Uniformity | Conv. Rate")
    print("|:---|:---:|:---:|:---:|")
    for dataset, kernel_data in data.items():
        ffm_data = kernel_data.get('none', {})
        parts = [DATASET_DISPLAY_NAMES.get(dataset, dataset)]
        for metric in ['nfe', 'uniformity', 'convergence_rate']:
            ffm_value = ffm_data.get(metric)
            _, best_kffm_value = find_best_kffm_value(kernel_data, metric, KFFM_KERNELS, lower_is_better=True)
            if ffm_value and best_kffm_value:
                if metric == 'nfe':
                    parts.append(f"{ffm_value.mean:.1f}/{best_kffm_value.mean:.1f}")
                elif metric == 'uniformity':
                    parts.append(f"{ffm_value.mean:.1f}/{best_kffm_value.mean:.1f}")
                else:
                    parts.append(f"{ffm_value.mean:.4f}/{best_kffm_value.mean:.4f}")
            else:
                parts.append("--/--")
        print("| " + " | ".join(parts) + " |")
    
    # Save markdown too
    md_path = output_path.parent / 'metrics_table.md'
    with open(md_path, 'w') as f:
        f.write("# Metrics Comparison Tables\n\n")
        f.write("NFE = Number of Function Evaluations (lower is better)\n")
        f.write("Uniformity = Velocity Uniformity / Time Variance (lower is better)\n")
        f.write("Conv = Convergence Rate (lower is better)\n\n")
        f.write("## Main Table (FFM vs best k-FFM)\n\n")
        f.write("See `metrics_table_main.tex`\n\n")
        f.write("## Full Table (All Kernels)\n\n")
        f.write("See `metrics_table_appendix.tex`\n")
    print(f"\n✓ Saved Markdown: {md_path}")


if __name__ == '__main__':
    main()
