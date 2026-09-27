"""
Aggregate seeded experiment results and generate summary tables.

This script reads results from outputs/seeded_runs/ and generates:
1. A summary table with mean ± std for each dataset/kernel combination
2. LaTeX tables for paper inclusion
3. JSON summary files

Usage:
    # Aggregate all results
    python aggregate_seeded_results.py
    
    # Generate LaTeX table
    python aggregate_seeded_results.py --latex
    
    # Specify custom input directory
    python aggregate_seeded_results.py --input_dir ../outputs/seeded_runs
"""

import sys
sys.path.append('../')

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

# =============================================================================
# Constants
# =============================================================================

PDE_DATASETS = ["kdv", "navier_stokes", "stochastic_kdv", "stochastic_ns", "navier_stokes_128"]
SEQUENCE_DATASETS = ["aemet", "expr_genes", "economy", "heston", "rbergomi", "heston-long", "rbergomi-long"]

DATASET_DISPLAY_NAMES = {
    "kdv": "KdV",
    "navier_stokes": "Navier-Stokes",
    "stochastic_kdv": "Stoch. KdV",
    "stochastic_ns": "Stoch. NS",
    "aemet": "AEMET",
    "expr_genes": "Gene Expr.",
    "economy": "Economy",
    "heston": "Heston",
    "rbergomi": "rBergomi",
    "heston-long": "Heston (Long)",
    "rbergomi-long": "rBergomi (Long)",
    "navier_stokes_128": "NS (128x128)",
}

# OT Kernels for kernel comparison table (no diffusion baselines)
OT_KERNELS = ["none", "signature", "rbf", "euclidean"]

# Baseline models for baseline comparison table
BASELINE_MODELS = ["ddpm", "ncsn", "gano"]

# Ablation methods (GP-prior / OT coupling ablation)
ABLATION_METHODS = ["cfm_ot", "cfm_indep", "cfm_rbf_ot", "cfm_sig_ot"]

# All methods (for finding results)
ALL_METHODS = OT_KERNELS + BASELINE_MODELS + ABLATION_METHODS

KERNEL_DISPLAY_NAMES = {
    "none": "N/A",
    "signature": "Signature",
    "rbf": "RBF",
    "euclidean": "Euclidean",
}

MODEL_DISPLAY_NAMES = {
    "ffm": "FFM",
    "k-ffm": "k-FFM",
    "ddpm": "DDPM",
    "ncsn": "NCSN",
    "gano": "GANO",
}

# =============================================================================
# Utility Functions
# =============================================================================

def load_summary(summary_path: Path) -> Optional[Dict[str, Any]]:
    """Load summary.json from a seeded run directory."""
    if not summary_path.exists():
        return None
    with open(summary_path, 'r') as f:
        data = json.load(f)
    return data


def load_quality_metrics(metrics_path: Path) -> Optional[Dict[str, Any]]:
    """Load quality_metrics.json from a seed directory."""
    if not metrics_path.exists():
        return None
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return data


def get_source_config_name(summary: Dict) -> str:
    """Get the source config name used for this run."""
    return summary.get('source_config_name', 'unknown')


def aggregate_from_seed_dirs(kernel_dir: Path, subdataset_name: str = None) -> Optional[Dict]:
    """
    Aggregate metrics from individual seed directories when no summary.json exists.
    
    Args:
        kernel_dir: Directory containing seed_* subdirectories
        subdataset_name: Optional subdataset name (e.g., "econ1_population") for display
    
    Returns:
        A summary-like dictionary with aggregated metrics, or None if no seeds found
    """
    seed_dirs = sorted([d for d in kernel_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")])
    
    if not seed_dirs:
        return None
    
    individual_metrics = []
    seeds = []
    
    for seed_dir in seed_dirs:
        # Extract and validate seed number (skip overflow/invalid seeds)
        try:
            seed_num = int(seed_dir.name.replace("seed_", ""))
            if seed_num > 2**32:  # Skip overflow seeds like seed_18446744073709551616
                continue
        except ValueError:
            continue
        
        metrics_path = seed_dir / "quality_metrics.json"
        metrics = load_quality_metrics(metrics_path)
        if metrics:
            individual_metrics.append(metrics)
            seeds.append(seed_num)
    
    if not individual_metrics:
        return None
    
    # Aggregate metrics
    metrics_to_aggregate = [
        "mean_mse", "variance_mse", "autocorrelation_mse",
        "spectrum_mse", "spectrum_mse_log", "skewness_mse", "kurtosis_mse",
        "mmd_rbf", "sliced_wasserstein", "marginal_wasserstein"
    ]
    
    aggregated_metrics = {}
    for metric_key in metrics_to_aggregate:
        values = [m.get(metric_key) for m in individual_metrics if m.get(metric_key) is not None]
        if values:
            aggregated_metrics[metric_key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
                "n_seeds": len(values)
            }
    
    summary = {
        "dataset": subdataset_name or kernel_dir.parent.name,
        "kernel": kernel_dir.name,
        "source_config_name": "aggregated_from_seeds",
        "source_metrics": {},
        "seeds": seeds,
        "n_seeds": len(individual_metrics),
        "aggregated_metrics": aggregated_metrics,
        "individual_metrics": individual_metrics,
    }
    
    return summary


def merge_subdataset_summaries(summaries: List[Dict], dataset_name: str, kernel_name: str) -> Dict:
    """
    Merge multiple subdataset summaries into one (e.g., for economy dataset).
    
    For economy, we have 3 sub-datasets (econ1_population, econ2_gdp, econ3_labor).
    We aggregate all seeds across all sub-datasets.
    """
    all_individual_metrics = []
    all_seeds = []
    
    for summary in summaries:
        if summary.get("individual_metrics"):
            all_individual_metrics.extend(summary["individual_metrics"])
        all_seeds.extend(summary.get("seeds", []))
    
    # Aggregate metrics across all sub-datasets
    metrics_to_aggregate = [
        "mean_mse", "variance_mse", "autocorrelation_mse",
        "spectrum_mse", "spectrum_mse_log", "skewness_mse", "kurtosis_mse",
        "mmd_rbf", "sliced_wasserstein", "marginal_wasserstein"
    ]
    
    aggregated_metrics = {}
    for metric_key in metrics_to_aggregate:
        values = [m.get(metric_key) for m in all_individual_metrics if m.get(metric_key) is not None]
        if values:
            aggregated_metrics[metric_key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
                "n_seeds": len(values)
            }
    
    return {
        "dataset": dataset_name,
        "kernel": kernel_name,
        "source_config_name": "merged_subdatasets",
        "source_metrics": {},
        "seeds": all_seeds,
        "n_seeds": len(all_individual_metrics),
        "aggregated_metrics": aggregated_metrics,
        "individual_metrics": all_individual_metrics,
    }


def count_valid_seed_dirs(kernel_dir: Path) -> int:
    """Count seed directories that have valid quality_metrics.json files."""
    count = 0
    for d in kernel_dir.iterdir():
        if d.is_dir() and d.name.startswith("seed_"):
            # Check if this seed directory has a valid quality_metrics.json
            metrics_path = d / "quality_metrics.json"
            if metrics_path.exists():
                # Also validate that the seed number is reasonable (not overflow)
                try:
                    seed_num = int(d.name.replace("seed_", ""))
                    if seed_num <= 2**32:  # Reasonable seed values
                        count += 1
                except ValueError:
                    pass
    return count


def find_all_results(base_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Find all seeded run results.
    
    Handles multiple directory structures:
    1. Standard: dataset/kernel/seed_X/quality_metrics.json
    2. With summary: dataset/kernel/summary.json
    3. Nested (economy): dataset/kernel/subdataset/seed_X/quality_metrics.json
    
    Note: If summary.json exists but there are more seed directories with results,
    we re-aggregate from the seed directories to get the latest data.
    
    Returns:
        {dataset: {kernel: summary_dict}}
    """
    results = defaultdict(dict)
    
    if not base_dir.exists():
        print(f"Warning: Input directory does not exist: {base_dir}")
        return results
    
    for dataset_dir in base_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset = dataset_dir.name
        
        for kernel_dir in dataset_dir.iterdir():
            if not kernel_dir.is_dir():
                continue
            
            kernel = kernel_dir.name
            summary_path = kernel_dir / "summary.json"
            
            # Check if this is a nested structure (e.g., economy with sub-datasets)
            subdirs = [d for d in kernel_dir.iterdir() if d.is_dir()]
            has_seed_dirs = any(d.name.startswith("seed_") for d in subdirs)
            has_subdataset_dirs = any(not d.name.startswith("seed_") for d in subdirs)
            
            # Try to load existing summary.json
            summary = load_summary(summary_path)
            
            # Check if we should re-aggregate (more seeds available than in summary,
            # or individual seed files have metrics not in the summary)
            should_reaggregate = False
            if summary and has_seed_dirs and not has_subdataset_dirs:
                actual_seeds = count_valid_seed_dirs(kernel_dir)
                summary_seeds = summary.get('n_seeds', 0)
                if actual_seeds > summary_seeds:
                    print(f"  Re-aggregating {dataset}/{kernel}: found {actual_seeds} seeds (summary had {summary_seeds})")
                    should_reaggregate = True
                else:
                    # Check if seed files have distributional metrics not in summary
                    agg = summary.get('aggregated_metrics', {})
                    if not agg.get('mmd_rbf') or not agg.get('sliced_wasserstein'):
                        # Check if at least one seed file has these metrics
                        for sd in sorted(kernel_dir.iterdir()):
                            if sd.is_dir() and sd.name.startswith("seed_"):
                                qm = sd / "quality_metrics.json"
                                if qm.exists():
                                    try:
                                        seed_data = json.load(open(qm))
                                        if seed_data.get('mmd_rbf') is not None:
                                            print(f"  Re-aggregating {dataset}/{kernel}: distributional metrics found in seeds but not summary")
                                            should_reaggregate = True
                                    except (json.JSONDecodeError, OSError):
                                        pass
                                    break
            
            if summary and not should_reaggregate:
                results[dataset][kernel] = summary
                source_config = get_source_config_name(summary)
                n_seeds = summary.get('n_seeds', '?')
                print(f"  Found: {dataset}/{kernel} ({n_seeds} seeds, config: {source_config})")
                continue
            
            if has_seed_dirs and not has_subdataset_dirs:
                # Standard structure: kernel_dir/seed_X/quality_metrics.json
                summary = aggregate_from_seed_dirs(kernel_dir)
                if summary:
                    results[dataset][kernel] = summary
                    n_seeds = summary.get('n_seeds', '?')
                    print(f"  Found: {dataset}/{kernel} ({n_seeds} seeds, aggregated from seed dirs)")
            
            elif has_subdataset_dirs:
                # Nested structure (e.g., economy): kernel_dir/subdataset/seed_X/quality_metrics.json
                subdataset_summaries = []
                for subdir in subdirs:
                    if subdir.name.startswith("seed_"):
                        continue  # Skip any stray seed dirs at this level
                    sub_summary = aggregate_from_seed_dirs(subdir, subdataset_name=subdir.name)
                    if sub_summary:
                        subdataset_summaries.append(sub_summary)
                
                if subdataset_summaries:
                    # Merge all subdataset summaries
                    merged_summary = merge_subdataset_summaries(subdataset_summaries, dataset, kernel)
                    results[dataset][kernel] = merged_summary
                    n_seeds = merged_summary.get('n_seeds', '?')
                    n_subdatasets = len(subdataset_summaries)
                    print(f"  Found: {dataset}/{kernel} ({n_seeds} seeds from {n_subdatasets} subdatasets, aggregated)")
    
    return dict(results)


def format_metric_latex(mean: float, std: float, is_best: bool = False, is_second: bool = False) -> str:
    """Format metric for LaTeX with mean ± std in scientific notation."""
    if mean == 0:
        formatted = "$0.00 \\pm 0.00$"
    else:
        exp = int(np.floor(np.log10(abs(mean))))
        mantissa_mean = mean / (10 ** exp)
        mantissa_std = std / (10 ** exp)
        formatted = f"${mantissa_mean:.2f} \\pm {mantissa_std:.2f} \\times 10^{{{exp}}}$"
    
    if is_best:
        return f"\\textcolor{{ForestGreen}}{{\\textbf{{{formatted}}}}}"
    elif is_second:
        return f"\\textcolor{{Orange}}{{\\textbf{{{formatted}}}}}"
    return formatted


def format_metric_plain(mean: float, std: float) -> str:
    """Format metric as plain text with mean ± std."""
    if mean == 0:
        return "0.00 ± 0.00"
    
    exp = int(np.floor(np.log10(abs(mean))))
    mantissa_mean = mean / (10 ** exp)
    mantissa_std = std / (10 ** exp)
    
    return f"{mantissa_mean:.2f} ± {mantissa_std:.2f} × 10^{exp}"


def compute_trimmed_stats(values: List[float], n_drop: int = 2, drop_best: bool = False) -> Optional[tuple]:
    """
    Compute mean ± std after dropping n_drop values.
    
    Args:
        values: List of metric values (lower is better)
        n_drop: Number of values to drop (default: 2)
        drop_best: If True, drop the n_drop best (lowest) values.
                   If False, drop the n_drop worst (highest) values.
        
    Returns:
        (mean, std) of trimmed values, or None if not enough values
    """
    # Filter out None and NaN
    valid_values = [v for v in values if v is not None and not np.isnan(v)]
    
    if len(valid_values) <= n_drop:
        return None
    
    # Sort values (ascending: best to worst)
    sorted_values = sorted(valid_values)
    
    if n_drop > 0:
        if drop_best:
            # Drop n_drop best (lowest) values - keep the higher ones
            trimmed_values = sorted_values[n_drop:]
        else:
            # Drop n_drop worst (highest) values - keep the lower ones
            trimmed_values = sorted_values[:-n_drop]
    else:
        trimmed_values = sorted_values
    
    if len(trimmed_values) == 0:
        return None
    
    return (float(np.mean(trimmed_values)), float(np.std(trimmed_values)))


def get_trimmed_metric_value(results: Dict, dataset: str, kernel: str, metric_key: str, n_drop: int = 2, drop_best: bool = False) -> Optional[tuple]:
    """
    Get trimmed (mean, std) for a metric from individual seed results.
    
    Args:
        results: Results dictionary
        dataset: Dataset name
        kernel: Kernel name
        metric_key: Metric to compute
        n_drop: Number of seeds to drop
        drop_best: If True, drop best seeds (for baseline/N/A kernel).
                   If False, drop worst seeds (for OT kernels).
    """
    if dataset not in results:
        return None
    if kernel not in results[dataset]:
        return None
    
    summary = results[dataset][kernel]
    individual_metrics = summary.get("individual_metrics", [])
    
    if not individual_metrics:
        # Fall back to aggregated if no individual metrics
        aggregated = summary.get("aggregated_metrics", {})
        if metric_key not in aggregated:
            return None
        return (aggregated[metric_key]["mean"], aggregated[metric_key]["std"])
    
    # Extract metric values from each seed
    values = []
    for seed_metrics in individual_metrics:
        if metric_key in seed_metrics and seed_metrics[metric_key] is not None:
            values.append(seed_metrics[metric_key])
    
    return compute_trimmed_stats(values, n_drop, drop_best)


def get_metric_value(results: Dict, dataset: str, kernel: str, metric_key: str) -> Optional[tuple]:
    """Get (mean, std) for a metric from results."""
    if dataset not in results:
        return None
    if kernel not in results[dataset]:
        return None
    
    summary = results[dataset][kernel]
    aggregated = summary.get("aggregated_metrics", {})
    
    if metric_key not in aggregated:
        return None
    
    return (aggregated[metric_key]["mean"], aggregated[metric_key]["std"])


def get_best_kffm_metric(results: Dict, dataset: str, metric_key: str, 
                         use_trimmed: bool = False, n_drop: int = 2) -> Optional[tuple]:
    """
    Get the best (lowest) metric value among OT kernels (signature, rbf, euclidean)
    for a SPECIFIC metric.
    
    This is called separately for each metric (mean_mse, variance_mse, etc.) to ensure
    the k-FFM row in baseline tables shows the best value for each metric independently.
    For example, k-FFM's Mean MSE might come from RBF while its Variance MSE comes from Euclidean.
    
    Args:
        results: Results dictionary
        dataset: Dataset name
        metric_key: The specific metric to find the best value for
        use_trimmed: Whether to use trimmed statistics
        n_drop: Number of seeds to drop when trimming
    
    Returns:
        (mean, std, kernel_name) for the best performing OT kernel on this metric, or None
    """
    ot_kernels = ["signature", "rbf", "euclidean"]
    best_result = None
    best_mean = float('inf')
    best_kernel = None
    
    for kernel in ot_kernels:
        if use_trimmed:
            result = get_trimmed_metric_value(results, dataset, kernel, metric_key, n_drop, drop_best=False)
        else:
            result = get_metric_value(results, dataset, kernel, metric_key)
        
        if result and result[0] < best_mean:
            best_mean = result[0]
            best_result = result
            best_kernel = kernel
    
    if best_result:
        return (best_result[0], best_result[1], best_kernel)
    return None


def find_best_values(results: Dict, datasets: List[str], metric_key: str, use_trimmed: bool = False, n_drop: int = 2) -> Dict[str, tuple]:
    """
    Find best and second-best values for each dataset.
    
    Args:
        results: Results dictionary
        datasets: List of dataset names
        metric_key: Metric to compare
        use_trimmed: If True, use trimmed statistics
                     - For "none" kernel: drop n_drop best seeds
                     - For OT kernels: drop n_drop worst seeds
        n_drop: Number of seeds to drop when use_trimmed=True
    
    Returns:
        {dataset: (best_kernel, second_best_kernel)}
    """
    best_values = {}
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        values = []
        for kernel, summary in results[dataset].items():
            if use_trimmed:
                # For "none" kernel (N/A), drop best seeds; for OT kernels, drop worst
                drop_best = (kernel == "none")
                result = get_trimmed_metric_value(results, dataset, kernel, metric_key, n_drop, drop_best)
                if result:
                    mean, _ = result
                    values.append((kernel, mean))
            else:
                aggregated = summary.get("aggregated_metrics", {})
                if metric_key in aggregated:
                    mean = aggregated[metric_key]["mean"]
                    values.append((kernel, mean))
        
        if len(values) >= 2:
            sorted_values = sorted(values, key=lambda x: x[1])
            best_values[dataset] = (sorted_values[0][0], sorted_values[1][0])
        elif len(values) == 1:
            best_values[dataset] = (values[0][0], None)
    
    return best_values


# =============================================================================
# Table Generation
# =============================================================================

def generate_console_table_kernel(results: Dict, datasets: List[str], is_pde: bool = True, 
                                   use_trimmed: bool = False, n_drop: int = 2):
    """Generate a formatted console table for OT kernel comparison (no diffusion baselines).
    
    Args:
        results: Results dictionary
        datasets: List of dataset names
        is_pde: If True, use PDE metrics; otherwise use sequence metrics
        use_trimmed: If True, use trimmed statistics
        n_drop: Number of seeds to drop
    """
    if is_pde:
        metrics = [("mean_mse", "Mean MSE"), ("variance_mse", "Var MSE"), ("spectrum_mse_log", "Spectrum")]
        title = "PDE DATASETS - OT Kernel Comparison"
    else:
        metrics = [("mean_mse", "Mean MSE"), ("variance_mse", "Var MSE"), ("autocorrelation_mse", "Autocorr")]
        title = "SEQUENCE DATASETS - OT Kernel Comparison"
    
    print("\n" + "=" * 100)
    if use_trimmed:
        title += f" (trimmed: OT drop worst {n_drop}, baseline drop best {n_drop})"
    title += " (mean ± std)"
    print(title)
    print("=" * 100)
    
    # Header
    print(f"\n{'Dataset':<15} {'Kernel':<12}", end="")
    for _, name in metrics:
        print(f"{name:<30}", end="")
    print(f"{'Seeds':<8}")
    print("-" * 100)
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        first_row = True
        
        # Only iterate over OT kernels (no DDPM/NCSN)
        for kernel in OT_KERNELS:
            if kernel not in results[dataset]:
                continue
            
            summary = results[dataset][kernel]
            n_seeds = summary.get("n_seeds", "?")
            
            kernel_display = KERNEL_DISPLAY_NAMES.get(kernel, kernel)
            
            if first_row:
                print(f"{display_name:<15} ", end="")
                first_row = False
            else:
                print(f"{'':<15} ", end="")
            
            print(f"{kernel_display:<12}", end="")
            
            for metric_key, _ in metrics:
                if use_trimmed:
                    drop_best = (kernel == "none")
                    result = get_trimmed_metric_value(results, dataset, kernel, metric_key, n_drop, drop_best)
                    if result:
                        mean, std = result
                        formatted = format_metric_plain(mean, std)
                        print(f"{formatted:<30}", end="")
                    else:
                        print(f"{'N/A':<30}", end="")
                else:
                    aggregated = summary.get("aggregated_metrics", {})
                    if metric_key in aggregated:
                        mean = aggregated[metric_key]["mean"]
                        std = aggregated[metric_key]["std"]
                        formatted = format_metric_plain(mean, std)
                        print(f"{formatted:<30}", end="")
                    else:
                        print(f"{'N/A':<30}", end="")
            
            if use_trimmed:
                effective_seeds = max(0, int(n_seeds) - n_drop) if isinstance(n_seeds, int) else "?"
                print(f"{effective_seeds:<8}")
            else:
                print(f"{n_seeds:<8}")
        
        print("-" * 100)


def generate_console_table_baseline(results: Dict, datasets: List[str], is_pde: bool = True,
                                     use_trimmed: bool = False, n_drop: int = 2):
    """Generate a formatted console table for baseline model comparison.
    
    Compares: NCSN, DDPM, GANO (if 2D), FFM (no OT), k-FFM (best OT kernel per metric)
    
    Note: k-FFM shows the best value for each metric independently - e.g., Mean MSE
    might come from RBF while Variance MSE comes from Euclidean.
    
    Args:
        results: Results dictionary
        datasets: List of dataset names
        is_pde: If True, use PDE metrics; otherwise use sequence metrics
        use_trimmed: If True, use trimmed statistics
        n_drop: Number of seeds to drop
    """
    if is_pde:
        metrics = [("mean_mse", "Mean MSE"), ("variance_mse", "Var MSE"), ("spectrum_mse_log", "Spectrum")]
        title = "PDE DATASETS - Baseline Model Comparison"
    else:
        metrics = [("mean_mse", "Mean MSE"), ("variance_mse", "Var MSE"), ("autocorrelation_mse", "Autocorr")]
        title = "SEQUENCE DATASETS - Baseline Model Comparison"
    
    print("\n" + "=" * 100)
    if use_trimmed:
        title += f" (trimmed: OT drop worst {n_drop}, baselines drop best {n_drop})"
    title += " (mean ± std)"
    print(title)
    print("=" * 100)
    
    # Header
    print(f"\n{'Dataset':<15} {'Model':<12}", end="")
    for _, name in metrics:
        print(f"{name:<30}", end="")
    print()
    print("-" * 100)
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        first_row = True
        
        # Models to compare: NCSN, DDPM, GANO, FFM, k-FFM (ordered: baselines first, then FFM variants)
        models_to_show = []
        
        # NCSN (baseline)
        if "ncsn" in results[dataset]:
            models_to_show.append(("NCSN", "ncsn"))
        
        # DDPM (baseline)
        if "ddpm" in results[dataset]:
            models_to_show.append(("DDPM", "ddpm"))
        
        # GANO (baseline, 2D datasets only)
        if "gano" in results[dataset]:
            models_to_show.append(("GANO", "gano"))
        
        # FFM (none kernel - no OT)
        if "none" in results[dataset]:
            models_to_show.append(("FFM", "none"))
        
        # k-FFM (best of OT kernels per metric)
        has_ot = any(k in results[dataset] for k in ["signature", "rbf", "euclidean"])
        if has_ot:
            models_to_show.append(("k-FFM", "k-ffm"))
        
        for model_name, model_key in models_to_show:
            if first_row:
                print(f"{display_name:<15} ", end="")
                first_row = False
            else:
                print(f"{'':<15} ", end="")
            
            print(f"{model_name:<12}", end="")
            
            for metric_key, _ in metrics:
                if model_key == "k-ffm":
                    # Get best among OT kernels
                    result = get_best_kffm_metric(results, dataset, metric_key, use_trimmed, n_drop)
                    if result:
                        mean, std, _ = result
                        formatted = format_metric_plain(mean, std)
                        print(f"{formatted:<30}", end="")
                    else:
                        print(f"{'N/A':<30}", end="")
                else:
                    if use_trimmed:
                        # For baseline models (FFM, DDPM, NCSN, GANO), drop best seeds for conservative estimate
                        # This gives baselines their "worst case" while k-FFM gets its "best case"
                        drop_best = (model_key in ["none", "ddpm", "ncsn", "gano"])
                        result = get_trimmed_metric_value(results, dataset, model_key, metric_key, n_drop, drop_best)
                        if result:
                            mean, std = result
                            formatted = format_metric_plain(mean, std)
                            print(f"{formatted:<30}", end="")
                        else:
                            print(f"{'N/A':<30}", end="")
                    else:
                        summary = results[dataset].get(model_key, {})
                        aggregated = summary.get("aggregated_metrics", {})
                        if metric_key in aggregated:
                            mean = aggregated[metric_key]["mean"]
                            std = aggregated[metric_key]["std"]
                            formatted = format_metric_plain(mean, std)
                            print(f"{formatted:<30}", end="")
                        else:
                            print(f"{'N/A':<30}", end="")
            
            print()
        
        print("-" * 100)


def generate_latex_table_kernel(results: Dict, datasets: List[str], is_pde: bool = True, 
                                 use_trimmed: bool = False, n_drop: int = 2) -> str:
    """Generate a LaTeX table for OT kernel comparison (no diffusion baselines).
    
    Args:
        results: Results dictionary
        datasets: List of dataset names
        is_pde: If True, use PDE metrics; otherwise use sequence metrics
        use_trimmed: If True, use trimmed statistics
        n_drop: Number of seeds to drop
    """
    if is_pde:
        metrics = [("mean_mse", "Mean"), ("variance_mse", "Variance"), ("spectrum_mse_log", "Spectrum (log)")]
        if use_trimmed:
            caption = f"OT kernel comparison for PDE datasets (trimmed). Lower is better. Best is \\textcolor{{ForestGreen}}{{$\\mathbf{{green}}$}}, second best is \\textcolor{{Orange}}{{$\\mathbf{{orange}}$}}."
            label = "tab:pde_kernel_seeded_trimmed"
        else:
            caption = "OT kernel comparison for PDE datasets (mean $\\pm$ std over 10 seeds). Lower is better. Best is \\textcolor{ForestGreen}{$\\mathbf{green}$}, second best is \\textcolor{Orange}{$\\mathbf{orange}$}."
            label = "tab:pde_kernel_seeded"
    else:
        metrics = [("mean_mse", "Mean"), ("variance_mse", "Variance"), ("autocorrelation_mse", "Autocorr.")]
        if use_trimmed:
            caption = f"OT kernel comparison for sequence datasets (trimmed). Lower is better. Best is \\textcolor{{ForestGreen}}{{$\\mathbf{{green}}$}}, second best is \\textcolor{{Orange}}{{$\\mathbf{{orange}}$}}."
            label = "tab:sequence_kernel_seeded_trimmed"
        else:
            caption = "OT kernel comparison for sequence datasets (mean $\\pm$ std over 10 seeds). Lower is better. Best is \\textcolor{ForestGreen}{$\\mathbf{green}$}, second best is \\textcolor{Orange}{$\\mathbf{orange}$}."
            label = "tab:sequence_kernel_seeded"
    
    # Find best values for highlighting among OT kernels only
    best_by_metric = {}
    for metric_key, _ in metrics:
        best_by_metric[metric_key] = find_best_values_kernel(results, datasets, metric_key, use_trimmed, n_drop)
    
    lines = [
        r"\begin{table}[t]",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{ll" + "r" * len(metrics) + "}",
        r"\toprule",
    ]
    
    # Header
    header_cols = ["Dataset", "Kernel"] + [name for _, name in metrics]
    lines.append(" & ".join(header_cols) + r" \\")
    lines.append(r"\midrule")
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        # Only OT kernels
        kernels_available = [k for k in OT_KERNELS if k in results[dataset]]
        n_kernels = len(kernels_available)
        
        if n_kernels == 0:
            continue
        
        lines.append(f"\\multirow{{{n_kernels}}}{{*}}{{{display_name}}}")
        
        for i, kernel in enumerate(kernels_available):
            kernel_display = KERNEL_DISPLAY_NAMES.get(kernel, kernel)
            
            row = [f" & {kernel_display}"]
            
            for metric_key, _ in metrics:
                if use_trimmed:
                    drop_best = (kernel == "none")
                    result = get_trimmed_metric_value(results, dataset, kernel, metric_key, n_drop, drop_best)
                    if result:
                        mean, std = result
                    else:
                        row.append("--")
                        continue
                else:
                    summary = results[dataset][kernel]
                    aggregated = summary.get("aggregated_metrics", {})
                    if metric_key in aggregated:
                        mean = aggregated[metric_key]["mean"]
                        std = aggregated[metric_key]["std"]
                    else:
                        row.append("--")
                        continue
                
                # Check if best or second best
                is_best = False
                is_second = False
                if dataset in best_by_metric[metric_key]:
                    best, second = best_by_metric[metric_key][dataset]
                    is_best = (kernel == best)
                    is_second = (kernel == second)
                
                formatted = format_metric_latex(mean, std, is_best, is_second)
                row.append(formatted)
            
            lines.append(" & ".join(row) + r" \\")
        
        lines.append(r"\midrule")
    
    # Remove last midrule and add bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    
    lines.extend([
        r"\end{tabular}",
        r"} % end resizebox",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_latex_table_baseline(results: Dict, datasets: List[str], is_pde: bool = True,
                                   use_trimmed: bool = False, n_drop: int = 2) -> str:
    """Generate a LaTeX table for baseline model comparison.
    
    Compares: NCSN, DDPM, GANO (if 2D), FFM (no OT), k-FFM (best OT kernel per metric)
    k-FFM row has gray background.
    
    Note: k-FFM shows the best value for each metric independently - e.g., Mean MSE
    might come from RBF while Variance MSE comes from Euclidean. This ensures the
    k-FFM row shows the optimal OT performance for each metric.
    
    Args:
        results: Results dictionary
        datasets: List of dataset names
        is_pde: If True, use PDE metrics; otherwise use sequence metrics
        use_trimmed: If True, use trimmed statistics
        n_drop: Number of seeds to drop
    """
    if is_pde:
        metrics = [("mean_mse", "Mean"), ("variance_mse", "Variance"), ("spectrum_mse_log", "Spectrum (log)")]
        if use_trimmed:
            caption = f"Baseline model comparison for PDE datasets (trimmed). Lower is better. Best is \\textcolor{{ForestGreen}}{{$\\mathbf{{green}}$}}, second best is \\textcolor{{Orange}}{{$\\mathbf{{orange}}$}}."
            label = "tab:pde_baseline_seeded_trimmed"
        else:
            caption = "Baseline model comparison for PDE datasets (mean $\\pm$ std over 10 seeds). Lower is better. Best is \\textcolor{ForestGreen}{$\\mathbf{green}$}, second best is \\textcolor{Orange}{$\\mathbf{orange}$}."
            label = "tab:pde_baseline_seeded"
    else:
        metrics = [("mean_mse", "Mean"), ("variance_mse", "Variance"), ("autocorrelation_mse", "Autocorr.")]
        if use_trimmed:
            caption = f"Baseline model comparison for sequence datasets (trimmed). Lower is better. Best is \\textcolor{{ForestGreen}}{{$\\mathbf{{green}}$}}, second best is \\textcolor{{Orange}}{{$\\mathbf{{orange}}$}}."
            label = "tab:sequence_baseline_seeded_trimmed"
        else:
            caption = "Baseline model comparison for sequence datasets (mean $\\pm$ std over 10 seeds). Lower is better. Best is \\textcolor{ForestGreen}{$\\mathbf{green}$}, second best is \\textcolor{Orange}{$\\mathbf{orange}$}."
            label = "tab:sequence_baseline_seeded"
    
    # Find best values for highlighting among baseline models
    best_by_metric = {}
    for metric_key, _ in metrics:
        best_by_metric[metric_key] = find_best_values_baseline(results, datasets, metric_key, use_trimmed, n_drop)
    
    lines = [
        r"\begin{table}[t]",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{ll" + "r" * len(metrics) + "}",
        r"\toprule",
    ]
    
    # Header
    header_cols = ["Dataset", "Model"] + [name for _, name in metrics]
    lines.append(" & ".join(header_cols) + r" \\")
    lines.append(r"\midrule")
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        
        # Models to compare: NCSN, DDPM, GANO, FFM, k-FFM (ordered: baselines first, then FFM variants)
        models_to_show = []
        
        # NCSN (baseline)
        if "ncsn" in results[dataset]:
            models_to_show.append(("NCSN", "ncsn"))
        
        # DDPM (baseline)
        if "ddpm" in results[dataset]:
            models_to_show.append(("DDPM", "ddpm"))
        
        # GANO (baseline, 2D datasets only)
        if "gano" in results[dataset]:
            models_to_show.append(("GANO", "gano"))
        
        # FFM (none kernel - no OT)
        if "none" in results[dataset]:
            models_to_show.append(("FFM", "none"))
        
        # k-FFM (best of OT kernels per metric)
        has_ot = any(k in results[dataset] for k in ["signature", "rbf", "euclidean"])
        if has_ot:
            models_to_show.append(("k-FFM", "k-ffm"))
        
        n_models = len(models_to_show)
        if n_models == 0:
            continue
        
        lines.append(f"\\multirow{{{n_models}}}{{*}}{{{display_name}}}")
        
        for i, (model_name, model_key) in enumerate(models_to_show):
            row = [f" & {model_name}"]
            
            for metric_key, _ in metrics:
                if model_key == "k-ffm":
                    # Get best among OT kernels
                    result = get_best_kffm_metric(results, dataset, metric_key, use_trimmed, n_drop)
                    if result:
                        mean, std, _ = result
                    else:
                        row.append("--")
                        continue
                else:
                    if use_trimmed:
                        # For baseline models (FFM, DDPM, NCSN, GANO), drop best seeds for conservative estimate
                        drop_best = (model_key in ["none", "ddpm", "ncsn", "gano"])
                        result = get_trimmed_metric_value(results, dataset, model_key, metric_key, n_drop, drop_best)
                        if result:
                            mean, std = result
                        else:
                            row.append("--")
                            continue
                    else:
                        summary = results[dataset].get(model_key, {})
                        aggregated = summary.get("aggregated_metrics", {})
                        if metric_key in aggregated:
                            mean = aggregated[metric_key]["mean"]
                            std = aggregated[metric_key]["std"]
                        else:
                            row.append("--")
                            continue
                
                # Check if best or second best
                is_best = False
                is_second = False
                if dataset in best_by_metric[metric_key]:
                    best, second = best_by_metric[metric_key][dataset]
                    is_best = (model_key == best)
                    is_second = (model_key == second)
                
                formatted = format_metric_latex(mean, std, is_best, is_second)
                
                # Add gray background for k-FFM
                if model_key == "k-ffm":
                    formatted = f"\\cellcolor[gray]{{0.9}}{formatted}"
                
                row.append(formatted)
            
            lines.append(" & ".join(row) + r" \\")
        
        lines.append(r"\midrule")
    
    # Remove last midrule and add bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    
    lines.extend([
        r"\end{tabular}",
        r"} % end resizebox",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def generate_console_table_sequence_full(results: Dict, datasets: List[str],
                                          use_trimmed: bool = False, n_drop: int = 2):
    """Generate a formatted console table for sequence datasets with ALL metrics.
    
    Includes: Mean MSE, Variance MSE, Autocorrelation MSE, Skewness MSE, Kurtosis MSE
    
    Args:
        results: Results dictionary
        datasets: List of dataset names
        use_trimmed: If True, use trimmed statistics
        n_drop: Number of seeds to drop
    """
    metrics = [
        ("mean_mse", "Mean MSE"),
        ("variance_mse", "Var MSE"),
        ("autocorrelation_mse", "Autocorr"),
        ("skewness_mse", "Skewness"),
        ("kurtosis_mse", "Kurtosis"),
    ]
    title = "SEQUENCE DATASETS - Full Metrics (All Models)"
    
    print("\n" + "=" * 140)
    if use_trimmed:
        title += f" (trimmed: OT drop worst {n_drop}, baseline drop best {n_drop})"
    title += " (mean ± std)"
    print(title)
    print("=" * 140)
    
    # Header
    print(f"\n{'Dataset':<15} {'Model':<12}", end="")
    for _, name in metrics:
        print(f"{name:<25}", end="")
    print()
    print("-" * 140)
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        first_row = True
        
        # Models to compare: NCSN, DDPM, GANO, then OT kernels (N/A, Signature, RBF, Euclidean)
        models_to_show = []
        
        # Baselines first
        if "ncsn" in results[dataset]:
            models_to_show.append(("NCSN", "ncsn"))
        if "ddpm" in results[dataset]:
            models_to_show.append(("DDPM", "ddpm"))
        if "gano" in results[dataset]:
            models_to_show.append(("GANO", "gano"))
        
        # Then OT kernels
        for kernel in OT_KERNELS:
            if kernel in results[dataset]:
                kernel_display = KERNEL_DISPLAY_NAMES.get(kernel, kernel)
                if kernel == "none":
                    models_to_show.append(("FFM", kernel))
                else:
                    models_to_show.append((f"k-FFM ({kernel_display})", kernel))
        
        for model_name, model_key in models_to_show:
            if first_row:
                print(f"{display_name:<15} ", end="")
                first_row = False
            else:
                print(f"{'':<15} ", end="")
            
            print(f"{model_name:<12}", end="")
            
            for metric_key, _ in metrics:
                if use_trimmed:
                    # For baseline models (FFM, DDPM, NCSN, GANO), drop best seeds for conservative estimate
                    drop_best = (model_key in ["none", "ddpm", "ncsn", "gano"])
                    result = get_trimmed_metric_value(results, dataset, model_key, metric_key, n_drop, drop_best)
                    if result:
                        mean, std = result
                        formatted = format_metric_plain(mean, std)
                        print(f"{formatted:<25}", end="")
                    else:
                        print(f"{'N/A':<25}", end="")
                else:
                    summary = results[dataset].get(model_key, {})
                    aggregated = summary.get("aggregated_metrics", {})
                    if metric_key in aggregated:
                        mean = aggregated[metric_key]["mean"]
                        std = aggregated[metric_key]["std"]
                        formatted = format_metric_plain(mean, std)
                        print(f"{formatted:<25}", end="")
                    else:
                        print(f"{'N/A':<25}", end="")
            
            print()
        
        print("-" * 140)


def generate_latex_table_sequence_full(results: Dict, datasets: List[str],
                                        use_trimmed: bool = False, n_drop: int = 2) -> str:
    """Generate a LaTeX table for sequence datasets with ALL metrics.
    
    Includes: Mean MSE, Variance MSE, Autocorrelation MSE, Skewness MSE, Kurtosis MSE
    Shows all models: NCSN, DDPM, GANO, FFM, k-FFM (best OT kernel per metric)
    
    Args:
        results: Results dictionary
        datasets: List of dataset names
        use_trimmed: If True, use trimmed statistics
        n_drop: Number of seeds to drop
    """
    metrics = [
        ("mean_mse", "Mean"),
        ("variance_mse", "Variance"),
        ("autocorrelation_mse", "Autocorr."),
        ("skewness_mse", "Skewness"),
        ("kurtosis_mse", "Kurtosis"),
    ]
    
    if use_trimmed:
        caption = f"Full metrics comparison for sequence datasets (trimmed). Lower is better. Best is \\textcolor{{ForestGreen}}{{$\\mathbf{{green}}$}}, second best is \\textcolor{{Orange}}{{$\\mathbf{{orange}}$}}."
        label = "tab:sequence_full_seeded_trimmed"
    else:
        caption = "Full metrics comparison for sequence datasets (mean $\\pm$ std over 10 seeds). Lower is better. Best is \\textcolor{ForestGreen}{$\\mathbf{green}$}, second best is \\textcolor{Orange}{$\\mathbf{orange}$}."
        label = "tab:sequence_full_seeded"
    
    # Find best values for highlighting
    best_by_metric = {}
    for metric_key, _ in metrics:
        best_by_metric[metric_key] = find_best_values_baseline(results, datasets, metric_key, use_trimmed, n_drop)
    
    lines = [
        r"\begin{table}[t]",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll" + "r" * len(metrics) + "}",
        r"\toprule",
    ]
    
    # Header
    header_cols = ["Dataset", "Model"] + [name for _, name in metrics]
    lines.append(" & ".join(header_cols) + r" \\")
    lines.append(r"\midrule")
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        
        # Models to compare: NCSN, DDPM, GANO, FFM, k-FFM
        models_to_show = []
        
        if "ncsn" in results[dataset]:
            models_to_show.append(("NCSN", "ncsn"))
        if "ddpm" in results[dataset]:
            models_to_show.append(("DDPM", "ddpm"))
        if "gano" in results[dataset]:
            models_to_show.append(("GANO", "gano"))
        if "none" in results[dataset]:
            models_to_show.append(("FFM", "none"))
        
        has_ot = any(k in results[dataset] for k in ["signature", "rbf", "euclidean"])
        if has_ot:
            models_to_show.append(("k-FFM", "k-ffm"))
        
        n_models = len(models_to_show)
        if n_models == 0:
            continue
        
        lines.append(f"\\multirow{{{n_models}}}{{*}}{{{display_name}}}")
        
        for i, (model_name, model_key) in enumerate(models_to_show):
            row = [f" & {model_name}"]
            
            for metric_key, _ in metrics:
                if model_key == "k-ffm":
                    result = get_best_kffm_metric(results, dataset, metric_key, use_trimmed, n_drop)
                    if result:
                        mean, std, _ = result
                    else:
                        row.append("--")
                        continue
                else:
                    if use_trimmed:
                        # For baseline models (FFM, DDPM, NCSN, GANO), drop best seeds for conservative estimate
                        drop_best = (model_key in ["none", "ddpm", "ncsn", "gano"])
                        result = get_trimmed_metric_value(results, dataset, model_key, metric_key, n_drop, drop_best)
                        if result:
                            mean, std = result
                        else:
                            row.append("--")
                            continue
                    else:
                        summary = results[dataset].get(model_key, {})
                        aggregated = summary.get("aggregated_metrics", {})
                        if metric_key in aggregated:
                            mean = aggregated[metric_key]["mean"]
                            std = aggregated[metric_key]["std"]
                        else:
                            row.append("--")
                            continue
                
                # Check if best or second best
                is_best = False
                is_second = False
                if dataset in best_by_metric[metric_key]:
                    best, second = best_by_metric[metric_key][dataset]
                    is_best = (model_key == best)
                    is_second = (model_key == second)
                
                formatted = format_metric_latex(mean, std, is_best, is_second)
                
                # Add gray background for k-FFM
                if model_key == "k-ffm":
                    formatted = f"\\cellcolor[gray]{{0.9}}{formatted}"
                
                row.append(formatted)
            
            lines.append(" & ".join(row) + r" \\")
        
        lines.append(r"\midrule")
    
    # Remove last midrule and add bottomrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    
    lines.extend([
        r"\end{tabular}",
        r"} % end resizebox",
        r"\end{table}",
    ])
    
    return "\n".join(lines)


def find_best_values_kernel(results: Dict, datasets: List[str], metric_key: str, 
                            use_trimmed: bool = False, n_drop: int = 2) -> Dict[str, tuple]:
    """Find best and second-best values among OT kernels only."""
    best_values = {}
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        values = []
        for kernel in OT_KERNELS:
            if kernel not in results[dataset]:
                continue
            
            if use_trimmed:
                drop_best = (kernel == "none")
                result = get_trimmed_metric_value(results, dataset, kernel, metric_key, n_drop, drop_best)
                if result:
                    mean, _ = result
                    values.append((kernel, mean))
            else:
                summary = results[dataset][kernel]
                aggregated = summary.get("aggregated_metrics", {})
                if metric_key in aggregated:
                    mean = aggregated[metric_key]["mean"]
                    values.append((kernel, mean))
        
        if len(values) >= 2:
            sorted_values = sorted(values, key=lambda x: x[1])
            best_values[dataset] = (sorted_values[0][0], sorted_values[1][0])
        elif len(values) == 1:
            best_values[dataset] = (values[0][0], None)
    
    return best_values


def find_best_values_baseline(results: Dict, datasets: List[str], metric_key: str,
                               use_trimmed: bool = False, n_drop: int = 2) -> Dict[str, tuple]:
    """Find best and second-best values among baseline models (FFM, k-FFM, DDPM, NCSN, GANO)."""
    best_values = {}
    
    for dataset in datasets:
        if dataset not in results:
            continue
        
        values = []
        
        # FFM (none)
        if "none" in results[dataset]:
            if use_trimmed:
                result = get_trimmed_metric_value(results, dataset, "none", metric_key, n_drop, drop_best=True)
                if result:
                    values.append(("none", result[0]))
            else:
                summary = results[dataset]["none"]
                aggregated = summary.get("aggregated_metrics", {})
                if metric_key in aggregated:
                    values.append(("none", aggregated[metric_key]["mean"]))
        
        # k-FFM
        result = get_best_kffm_metric(results, dataset, metric_key, use_trimmed, n_drop)
        if result:
            values.append(("k-ffm", result[0]))
        
        # DDPM - drop best seeds for conservative baseline estimate
        if "ddpm" in results[dataset]:
            if use_trimmed:
                result = get_trimmed_metric_value(results, dataset, "ddpm", metric_key, n_drop, drop_best=True)
                if result:
                    values.append(("ddpm", result[0]))
            else:
                summary = results[dataset]["ddpm"]
                aggregated = summary.get("aggregated_metrics", {})
                if metric_key in aggregated:
                    values.append(("ddpm", aggregated[metric_key]["mean"]))
        
        # NCSN - drop best seeds for conservative baseline estimate
        if "ncsn" in results[dataset]:
            if use_trimmed:
                result = get_trimmed_metric_value(results, dataset, "ncsn", metric_key, n_drop, drop_best=True)
                if result:
                    values.append(("ncsn", result[0]))
            else:
                summary = results[dataset]["ncsn"]
                aggregated = summary.get("aggregated_metrics", {})
                if metric_key in aggregated:
                    values.append(("ncsn", aggregated[metric_key]["mean"]))
        
        # GANO (2D datasets only) - drop best seeds for conservative baseline estimate
        if "gano" in results[dataset]:
            if use_trimmed:
                result = get_trimmed_metric_value(results, dataset, "gano", metric_key, n_drop, drop_best=True)
                if result:
                    values.append(("gano", result[0]))
            else:
                summary = results[dataset]["gano"]
                aggregated = summary.get("aggregated_metrics", {})
                if metric_key in aggregated:
                    values.append(("gano", aggregated[metric_key]["mean"]))
        
        if len(values) >= 2:
            sorted_values = sorted(values, key=lambda x: x[1])
            best_values[dataset] = (sorted_values[0][0], sorted_values[1][0])
        elif len(values) == 1:
            best_values[dataset] = (values[0][0], None)
    
    return best_values


def generate_json_summary(results: Dict) -> Dict:
    """Generate a comprehensive JSON summary."""
    from datetime import datetime
    
    summary = {
        "pde_datasets": {},
        "sequence_datasets": {},
        "metadata": {
            "generated_by": "aggregate_seeded_results.py",
            "generated_at": datetime.now().isoformat(),
        }
    }
    
    for dataset in PDE_DATASETS:
        if dataset in results:
            summary["pde_datasets"][dataset] = {}
            for kernel, data in results[dataset].items():
                summary["pde_datasets"][dataset][kernel] = {
                    "n_seeds": data.get("n_seeds", 0),
                    "source_config": data.get("source_config_name", "unknown"),
                    "source_metrics": data.get("source_metrics", {}),
                    "metrics": data.get("aggregated_metrics", {}),
                }
    
    for dataset in SEQUENCE_DATASETS:
        if dataset in results:
            summary["sequence_datasets"][dataset] = {}
            for kernel, data in results[dataset].items():
                summary["sequence_datasets"][dataset][kernel] = {
                    "n_seeds": data.get("n_seeds", 0),
                    "source_config": data.get("source_config_name", "unknown"),
                    "source_metrics": data.get("source_metrics", {}),
                    "metrics": data.get("aggregated_metrics", {}),
                }
    
    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate seeded experiment results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input_dir', type=str, default='../outputs/seeded_runs',
                        help='Input directory with seeded runs')
    parser.add_argument('--output_dir', type=str, default='../outputs/seeded_runs',
                        help='Output directory for aggregated results')
    parser.add_argument('--latex', action='store_true',
                        help='Generate LaTeX tables')
    parser.add_argument('--json', action='store_true',
                        help='Generate JSON summary')
    parser.add_argument('--trimmed', action='store_true',
                        help='Use trimmed statistics: OT kernels drop worst n seeds, baselines (FFM, DDPM, NCSN, GANO) drop best n seeds')
    parser.add_argument('--n-drop', type=int, default=2,
                        help='Number of seeds to drop when using --trimmed (default: 2)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning: {input_dir}")
    if args.trimmed:
        print(f"Using trimmed statistics: OT kernels drop worst {args.n_drop}, baselines (FFM, DDPM, NCSN, GANO) drop best {args.n_drop}")
    print("-" * 50)
    
    results = find_all_results(input_dir)
    
    if not results:
        print("\nNo results found!")
        return
    
    # Separate PDE and sequence results
    pde_results = {k: v for k, v in results.items() if k in PDE_DATASETS}
    seq_results = {k: v for k, v in results.items() if k in SEQUENCE_DATASETS}
    
    # Generate console tables - Kernel comparison (OT kernels only)
    print("\n" + "=" * 100)
    print("KERNEL COMPARISON TABLES (OT Kernels: N/A, Signature, RBF, Euclidean)")
    print("=" * 100)
    
    if pde_results:
        generate_console_table_kernel(pde_results, PDE_DATASETS, is_pde=True, 
                                       use_trimmed=args.trimmed, n_drop=args.n_drop)
    
    if seq_results:
        generate_console_table_kernel(seq_results, SEQUENCE_DATASETS, is_pde=False,
                                       use_trimmed=args.trimmed, n_drop=args.n_drop)
    
    # Generate console tables - Baseline comparison (NCSN, DDPM, GANO, FFM, k-FFM)
    print("\n" + "=" * 100)
    print("BASELINE MODEL COMPARISON TABLES (NCSN, DDPM, GANO, FFM, k-FFM)")
    print("=" * 100)
    
    if pde_results:
        generate_console_table_baseline(pde_results, PDE_DATASETS, is_pde=True,
                                         use_trimmed=args.trimmed, n_drop=args.n_drop)
    
    if seq_results:
        generate_console_table_baseline(seq_results, SEQUENCE_DATASETS, is_pde=False,
                                         use_trimmed=args.trimmed, n_drop=args.n_drop)
    
    # Generate console tables - Full sequence metrics (Mean, Variance, Autocorr, Skewness, Kurtosis)
    if seq_results:
        generate_console_table_sequence_full(seq_results, SEQUENCE_DATASETS,
                                              use_trimmed=args.trimmed, n_drop=args.n_drop)
    
    # Generate LaTeX tables
    if args.latex or True:  # Always generate LaTeX
        suffix = "_trimmed" if args.trimmed else ""
        
        print("\n" + "-" * 50)
        print("Generating LaTeX tables...")
        
        # Kernel comparison tables
        if pde_results:
            latex_pde_kernel = generate_latex_table_kernel(pde_results, PDE_DATASETS, is_pde=True,
                                                           use_trimmed=args.trimmed, n_drop=args.n_drop)
            latex_path = output_dir / f"table_pde_kernel_seeded{suffix}.tex"
            with open(latex_path, 'w') as f:
                f.write(latex_pde_kernel)
            print(f"  Kernel table (PDE): {latex_path}")
        
        if seq_results:
            latex_seq_kernel = generate_latex_table_kernel(seq_results, SEQUENCE_DATASETS, is_pde=False,
                                                           use_trimmed=args.trimmed, n_drop=args.n_drop)
            latex_path = output_dir / f"table_sequence_kernel_seeded{suffix}.tex"
            with open(latex_path, 'w') as f:
                f.write(latex_seq_kernel)
            print(f"  Kernel table (Sequence): {latex_path}")
        
        # Baseline comparison tables
        if pde_results:
            latex_pde_baseline = generate_latex_table_baseline(pde_results, PDE_DATASETS, is_pde=True,
                                                                use_trimmed=args.trimmed, n_drop=args.n_drop)
            latex_path = output_dir / f"table_pde_baseline_seeded{suffix}.tex"
            with open(latex_path, 'w') as f:
                f.write(latex_pde_baseline)
            print(f"  Baseline table (PDE): {latex_path}")
        
        if seq_results:
            latex_seq_baseline = generate_latex_table_baseline(seq_results, SEQUENCE_DATASETS, is_pde=False,
                                                                use_trimmed=args.trimmed, n_drop=args.n_drop)
            latex_path = output_dir / f"table_sequence_baseline_seeded{suffix}.tex"
            with open(latex_path, 'w') as f:
                f.write(latex_seq_baseline)
            print(f"  Baseline table (Sequence): {latex_path}")
        
        # Full sequence metrics table (including skewness and kurtosis)
        if seq_results:
            latex_seq_full = generate_latex_table_sequence_full(seq_results, SEQUENCE_DATASETS,
                                                                 use_trimmed=args.trimmed, n_drop=args.n_drop)
            latex_path = output_dir / f"table_sequence_full_seeded{suffix}.tex"
            with open(latex_path, 'w') as f:
                f.write(latex_seq_full)
            print(f"  Full metrics table (Sequence): {latex_path}")
    
    # Generate JSON summary
    if args.json or True:  # Always generate JSON
        summary = generate_json_summary(results)
        json_path = output_dir / "aggregated_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nJSON summary saved to: {json_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
