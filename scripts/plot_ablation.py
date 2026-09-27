"""
Plot ablation study results as clean line charts.

For each ablation axis, plots metric vs. hyperparameter value with
the Independent (FFM) baseline as a horizontal dashed line.

Usage:
    python plot_ablation.py
    python plot_ablation.py --datasets aemet heston
    python plot_ablation.py --output-dir ../outputs/figures/
"""

import sys
sys.path.append('../')

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────
DATASET_DISPLAY = {
    "aemet": "AEMET",
    "heston": "Heston",
    "rbergomi": "rBergomi",
    "economy": "Economy",
    "expr_genes": "Gene Expr.",
    "kdv": "KdV",
    "navier_stokes": "Navier-Stokes",
    "stochastic_kdv": "Stochastic KdV",
    "stochastic_ns": "Stochastic NS",
}

AXIS_DISPLAY = {
    "rbf_sigma": r"RBF Bandwidth $\sigma$",
    "sinkhorn_eps": r"Sinkhorn Regularization $\varepsilon$",
    "sig_dyadic": "Signature Dyadic Order",
    "sig_sigma": r"Signature Static Kernel $\sigma$",
    "euc_eps": r"Euclidean Sinkhorn $\varepsilon$",
}

METRIC_DISPLAY = {
    "mean_mse": "Mean MSE",
    "variance_mse": "Variance MSE",
    "autocorrelation_mse": "Autocorrelation MSE",
    "spectrum_mse_log": "Log-Spectral MSE",
    "mmd_rbf": "MMD (RBF)",
    "sliced_wasserstein": "Sliced Wasserstein",
    "marginal_wasserstein": "Marginal W1",
}

PDE_DATASETS = ["kdv", "navier_stokes", "stochastic_kdv", "stochastic_ns"]

# Hyperparameter values for x-axis (must match run_ablation_study.py)
AXIS_XVALUES = {
    "rbf_sigma": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "sinkhorn_eps": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    "sig_dyadic": [0, 1, 2, 3],
    "sig_sigma": [0.1, 0.5, 1.0, 2.0, 5.0],
    "euc_eps": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
}

# Config name patterns (must match run_ablation_study.py)
AXIS_CONFIG_NAMES = {
    "rbf_sigma": lambda v: f"rbf_sigma{v}",
    "sinkhorn_eps": lambda v: f"rbf_eps{v}",
    "sig_dyadic": lambda v: f"sig_order{v}",
    "sig_sigma": lambda v: f"sig_sigma{v}",
    "euc_eps": lambda v: f"euc_eps{v}",
}

AXIS_COLORS = {
    "rbf_sigma": "#d62728",
    "sinkhorn_eps": "#d62728",
    "sig_dyadic": "#9467bd",
    "sig_sigma": "#9467bd",
    "euc_eps": "#2ca02c",
}


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────
def load_ablation_results(base_dir: Path) -> Dict:
    """Load all ablation results.

    Returns: {dataset: {axis: {config_name: [metric_dicts_per_seed]}}}
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    abl_dir = base_dir / "ablation_study"
    if not abl_dir.exists():
        print(f"No ablation_study directory at {abl_dir}")
        return results

    for ds_dir in sorted(abl_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name

        for axis_dir in sorted(ds_dir.iterdir()):
            if not axis_dir.is_dir():
                continue
            axis = axis_dir.name

            for cfg_dir in sorted(axis_dir.iterdir()):
                if not cfg_dir.is_dir():
                    continue
                # Skip summary JSON files
                if cfg_dir.name.endswith('.json'):
                    continue
                config_name = cfg_dir.name

                for seed_dir in sorted(cfg_dir.iterdir()):
                    if not seed_dir.is_dir():
                        continue
                    qm_file = seed_dir / "quality_metrics.json"
                    if qm_file.exists():
                        with open(qm_file, 'r') as f:
                            metrics = json.load(f)
                        results[dataset][axis][config_name].append(metrics)

    return results


def get_metric_stats(
    results: Dict,
    dataset: str,
    axis: str,
    metric: str,
) -> Tuple[List[float], np.ndarray, np.ndarray, Optional[float], Optional[float]]:
    """Extract mean ± std for each hyperparam value, plus baseline.

    Returns: (x_values, means, stds, baseline_mean, baseline_std)
    """
    axis_data = results.get(dataset, {}).get(axis, {})
    if not axis_data:
        return [], np.array([]), np.array([]), None, None

    x_values = AXIS_XVALUES[axis]
    name_fn = AXIS_CONFIG_NAMES[axis]

    means = []
    stds = []
    valid_x = []

    for xv in x_values:
        cfg_name = name_fn(xv)
        seed_metrics = axis_data.get(cfg_name, [])
        vals = [m.get(metric) for m in seed_metrics if m.get(metric) is not None]
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            valid_x.append(xv)

    # Baseline (independent)
    baseline_metrics = axis_data.get("independent", [])
    baseline_vals = [m.get(metric) for m in baseline_metrics if m.get(metric) is not None]
    baseline_mean = np.mean(baseline_vals) if baseline_vals else None
    baseline_std = np.std(baseline_vals) if baseline_vals else None

    return valid_x, np.array(means), np.array(stds), baseline_mean, baseline_std


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────
def plot_ablation_axis(
    results: Dict,
    datasets: List[str],
    axis: str,
    metrics: List[str],
    output_dir: Path,
) -> None:
    """Plot one ablation axis: one subplot per dataset, each with metric lines."""

    n_ds = len(datasets)
    n_met = len(metrics)

    fig, axes_grid = plt.subplots(n_met, n_ds, figsize=(4.5 * n_ds, 3.5 * n_met),
                                  squeeze=False)

    has_any_data = False
    use_log_x = axis in ("sinkhorn_eps", "euc_eps")

    for col, dataset in enumerate(datasets):
        for row, metric in enumerate(metrics):
            ax = axes_grid[row][col]

            x_vals, means, stds, bl_mean, bl_std = get_metric_stats(
                results, dataset, axis, metric,
            )

            if len(x_vals) == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
                ax.set_visible(True)
                continue

            has_any_data = True
            color = AXIS_COLORS.get(axis, "#1f77b4")

            ax.plot(x_vals, means, 'o-', color=color, linewidth=1.5, markersize=5)
            ax.fill_between(x_vals, means - stds, means + stds, alpha=0.15, color=color)

            # Baseline dashed line
            if bl_mean is not None:
                ax.axhline(bl_mean, color='gray', linestyle='--', linewidth=1,
                           label='Independent (FFM)')
                if bl_std is not None and bl_std > 0:
                    ax.axhspan(bl_mean - bl_std, bl_mean + bl_std,
                               alpha=0.08, color='gray')

            if use_log_x:
                ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

            if row == n_met - 1:
                ax.set_xlabel(AXIS_DISPLAY.get(axis, axis), fontsize=10)
            if col == 0:
                ax.set_ylabel(METRIC_DISPLAY.get(metric, metric), fontsize=10)
            if row == 0:
                ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=11)

    if not has_any_data:
        plt.close(fig)
        return

    # Legend from first subplot that has data
    for ax_row in axes_grid:
        for ax in ax_row:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='upper center', ncol=2,
                           fontsize=9, bbox_to_anchor=(0.5, 1.02))
                break
        if handles:
            break

    fig.suptitle(f"Ablation: {AXIS_DISPLAY.get(axis, axis)}", y=1.06, fontsize=13)
    fig.tight_layout()

    out_path = output_dir / f"ablation_{axis}.pdf"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_combined_ablation(
    results: Dict,
    datasets: List[str],
    output_dir: Path,
) -> None:
    """Plot a compact 2-column figure: RBF sigma + Sinkhorn eps side by side,
    one row per dataset. Uses mean_mse as the metric."""

    metric = "mean_mse"
    axes_to_plot = ["rbf_sigma", "sinkhorn_eps"]
    avail_ds = [d for d in datasets if any(
        get_metric_stats(results, d, ax, metric)[1].size > 0
        for ax in axes_to_plot
    )]

    if not avail_ds:
        return

    n_ds = len(avail_ds)
    fig, axes_grid = plt.subplots(n_ds, 2, figsize=(10, 3 * n_ds), squeeze=False)

    for row, dataset in enumerate(avail_ds):
        for col, axis in enumerate(axes_to_plot):
            ax = axes_grid[row][col]
            x_vals, means, stds, bl_mean, bl_std = get_metric_stats(
                results, dataset, axis, metric,
            )

            if len(x_vals) == 0:
                continue

            color = AXIS_COLORS.get(axis, "#1f77b4")
            ax.plot(x_vals, means, 'o-', color=color, linewidth=1.5, markersize=5)
            ax.fill_between(x_vals, means - stds, means + stds, alpha=0.15, color=color)

            if bl_mean is not None:
                ax.axhline(bl_mean, color='gray', linestyle='--', linewidth=1,
                           label='Independent')
                if bl_std is not None and bl_std > 0:
                    ax.axhspan(bl_mean - bl_std, bl_mean + bl_std,
                               alpha=0.08, color='gray')

            if axis in ("sinkhorn_eps", "euc_eps"):
                ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

            if row == n_ds - 1:
                ax.set_xlabel(AXIS_DISPLAY.get(axis, axis), fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{DATASET_DISPLAY.get(dataset, dataset)}\n{METRIC_DISPLAY[metric]}",
                              fontsize=9)
            if row == 0:
                ax.set_title(AXIS_DISPLAY.get(axis, axis), fontsize=11)

    handles, labels = axes_grid[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2,
                   fontsize=9, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Hyperparameter Sensitivity (Mean MSE)", y=1.06, fontsize=13)
    fig.tight_layout()

    out_path = output_dir / "ablation_combined.pdf"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def generate_summary_table(results: Dict, datasets: List[str], output_dir: Path) -> None:
    """Generate markdown summary of ablation results."""

    lines = [
        "# Hyperparameter Ablation Results",
        "",
        "Each cell: mean +/- std across seeds. Bold = best per axis.",
        "",
    ]

    for dataset in datasets:
        ds_data = results.get(dataset, {})
        if not ds_data:
            continue

        ds_label = DATASET_DISPLAY.get(dataset, dataset)
        is_pde = dataset in PDE_DATASETS
        primary_metric = "mean_mse"

        lines.append(f"## {ds_label}")
        lines.append("")

        for axis in sorted(ds_data.keys()):
            x_values = AXIS_XVALUES.get(axis, [])
            if not x_values:
                continue

            axis_label = AXIS_DISPLAY.get(axis, axis)
            name_fn = AXIS_CONFIG_NAMES.get(axis)
            if not name_fn:
                continue

            lines.append(f"### {axis_label}")
            lines.append("")

            # Header
            header = "| Value | Mean MSE | Variance MSE |"
            sep = "|-------|:--------:|:------------:|"
            lines.append(header)
            lines.append(sep)

            # Baseline
            bl_metrics = ds_data[axis].get("independent", [])
            bl_vals_mean = [m.get("mean_mse") for m in bl_metrics if m.get("mean_mse") is not None]
            bl_vals_var = [m.get("variance_mse") for m in bl_metrics if m.get("variance_mse") is not None]
            if bl_vals_mean:
                bl_m = f"{np.mean(bl_vals_mean):.2e} +/- {np.std(bl_vals_mean):.2e}"
            else:
                bl_m = "N/A"
            if bl_vals_var:
                bl_v = f"{np.mean(bl_vals_var):.2e} +/- {np.std(bl_vals_var):.2e}"
            else:
                bl_v = "N/A"
            lines.append(f"| Independent | {bl_m} | {bl_v} |")

            for xv in x_values:
                cfg_name = name_fn(xv)
                seed_metrics = ds_data[axis].get(cfg_name, [])
                mean_vals = [m.get("mean_mse") for m in seed_metrics if m.get("mean_mse") is not None]
                var_vals = [m.get("variance_mse") for m in seed_metrics if m.get("variance_mse") is not None]

                if mean_vals:
                    cell_m = f"{np.mean(mean_vals):.2e} +/- {np.std(mean_vals):.2e}"
                else:
                    cell_m = "N/A"
                if var_vals:
                    cell_v = f"{np.mean(var_vals):.2e} +/- {np.std(var_vals):.2e}"
                else:
                    cell_v = "N/A"
                lines.append(f"| {xv} | {cell_m} | {cell_v} |")

            lines.append("")

    summary_path = output_dir / "ablation_summary.md"
    with open(summary_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved {summary_path}")


# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot ablation study results")
    parser.add_argument('--datasets', nargs='+', default=None,
                        help="Datasets to plot (default: all available)")
    parser.add_argument('--output-dir', type=str, default='../outputs/figures/',
                        help="Output directory for figures")
    parser.add_argument('--base-dir', type=str, default='../outputs/seeded_runs',
                        help="Base directory with ablation_study/ subdirectory")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_ablation_results(base_dir)
    if not results:
        print("No ablation study data found.")
        return

    datasets = args.datasets or sorted(results.keys())
    print(f"Datasets: {datasets}")

    # Determine metrics per dataset type
    for axis_name in AXIS_XVALUES:
        avail_ds = [d for d in datasets if axis_name in results.get(d, {})]
        if not avail_ds:
            continue

        # Choose metrics
        seq_ds = [d for d in avail_ds if d not in PDE_DATASETS]
        pde_ds = [d for d in avail_ds if d in PDE_DATASETS]

        if seq_ds:
            plot_ablation_axis(results, seq_ds, axis_name,
                               ["mean_mse", "variance_mse"], output_dir)
        if pde_ds:
            plot_ablation_axis(results, pde_ds, axis_name,
                               ["mean_mse", "variance_mse"], output_dir)

    # Combined figure
    plot_combined_ablation(results, datasets, output_dir)

    # Summary table
    generate_summary_table(results, datasets, output_dir)


if __name__ == '__main__':
    main()
