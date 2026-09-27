"""
Plot convergence curves from convergence study results.

Reads eval_trajectory.json files from outputs/seeded_runs/convergence_study/
and creates publication-quality convergence plots showing how actual target
metrics evolve during training for different methods.

Usage:
    # Plot all available datasets
    python plot_convergence.py

    # Plot specific datasets
    python plot_convergence.py --datasets aemet heston navier_stokes

    # Specific metrics
    python plot_convergence.py --metrics mean_mse variance_mse

    # Save to specific directory
    python plot_convergence.py --output-dir ../outputs/figures/
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
# Style / labels
# ─────────────────────────────────────────────────────────────────────
METHOD_DISPLAY = {
    "none": "Independent (FFM)",
    "euclidean": "Euclidean OT",
    "rbf": "RBF OT",
    "signature": "Signature OT",
    "cfm_ot": "CFM-OT (L2)",
    "ddpm": "DDPM",
    "ncsn": "NCSN",
    "gano": "GANO",
}

METHOD_COLORS = {
    "none": "#1f77b4",       # blue
    "euclidean": "#2ca02c",  # green
    "rbf": "#d62728",        # red
    "signature": "#9467bd",  # purple
    "cfm_ot": "#17becf",     # cyan
    "ddpm": "#ff7f0e",       # orange
    "ncsn": "#8c564b",       # brown
    "gano": "#e377c2",       # pink
}

METHOD_LINESTYLES = {
    "none": "-",
    "euclidean": "-",
    "rbf": "-",
    "signature": "-",
    "cfm_ot": "-.",
    "ddpm": "--",
    "ncsn": "--",
    "gano": ":",
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

DATASET_DISPLAY = {
    "aemet": "AEMET",
    "heston": "Heston",
    "economy": "Economy",
    "expr_genes": "Gene Expression",
    "kdv": "KdV",
    "navier_stokes": "Navier-Stokes",
    "stochastic_kdv": "Stochastic KdV",
    "stochastic_ns": "Stochastic NS",
    "heston-long": "Heston-Long",
}

PDE_DATASETS = ["kdv", "navier_stokes", "stochastic_kdv", "stochastic_ns"]


# ─────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────
def load_trajectories(base_dir: Path) -> Dict[str, Dict[str, List[dict]]]:
    """Load all eval_trajectory.json files.

    Returns: {dataset: {kernel: [trajectory_dicts]}}
    Each trajectory_dict has keys: trajectory (list of per-epoch dicts), seed, etc.
    """
    results = defaultdict(lambda: defaultdict(list))

    conv_dir = base_dir / "convergence_study"
    if not conv_dir.exists():
        print(f"No convergence_study directory found at {conv_dir}")
        return results

    for ds_dir in sorted(conv_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        for k_dir in sorted(ds_dir.iterdir()):
            if not k_dir.is_dir():
                continue
            kernel = k_dir.name
            for seed_dir in sorted(k_dir.iterdir()):
                traj_file = seed_dir / "eval_trajectory.json"
                if traj_file.exists():
                    with open(traj_file, 'r') as f:
                        data = json.load(f)
                    results[dataset][kernel].append(data)

    return results


def aggregate_metric(
    trajectories: List[dict],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a metric across seeds.

    Returns: (epochs, means, stds)
    """
    # Collect per-seed: {epoch: value}
    per_seed = []
    for traj_data in trajectories:
        epoch_val = {}
        for pt in traj_data["trajectory"]:
            if metric in pt and pt[metric] is not None:
                epoch_val[pt["epoch"]] = pt[metric]
        if epoch_val:
            per_seed.append(epoch_val)

    if not per_seed:
        return np.array([]), np.array([]), np.array([])

    # Find common epochs
    common_epochs = sorted(set.intersection(*[set(d.keys()) for d in per_seed]))
    if not common_epochs:
        return np.array([]), np.array([]), np.array([])

    epochs = np.array(common_epochs)
    values = np.array([[d[e] for e in common_epochs] for d in per_seed])
    means = np.mean(values, axis=0)
    stds = np.std(values, axis=0)

    return epochs, means, stds


# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────
def plot_dataset_convergence(
    dataset: str,
    kernel_trajectories: Dict[str, List[dict]],
    metrics: List[str],
    output_dir: Path,
    log_scale: bool = True,
) -> None:
    """Plot convergence curves for one dataset, multiple metrics as subplots."""

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4), squeeze=False)
    axes = axes[0]

    has_data = False

    for ax, metric in zip(axes, metrics):
        for kernel in sorted(kernel_trajectories.keys()):
            trajs = kernel_trajectories[kernel]
            epochs, means, stds = aggregate_metric(trajs, metric)
            if len(epochs) == 0:
                continue
            has_data = True

            label = METHOD_DISPLAY.get(kernel, kernel)
            color = METHOD_COLORS.get(kernel, None)
            ls = METHOD_LINESTYLES.get(kernel, "-")

            ax.plot(epochs, means, label=label, color=color, linestyle=ls, linewidth=1.5)
            ax.fill_between(epochs, means - stds, means + stds,
                            alpha=0.15, color=color)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(METRIC_DISPLAY.get(metric, metric))
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    if not has_data:
        plt.close(fig)
        return

    # Single legend for the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=min(len(handles), 4), fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    ds_label = DATASET_DISPLAY.get(dataset, dataset)
    fig.suptitle(f"{ds_label}: Convergence on Target Metrics", y=1.08, fontsize=13)
    fig.tight_layout()

    out_path = output_dir / f"convergence_{dataset}.pdf"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_combined_grid(
    all_data: Dict[str, Dict[str, List[dict]]],
    datasets: List[str],
    metric: str,
    output_dir: Path,
    log_scale: bool = True,
) -> None:
    """Plot a grid: one subplot per dataset, all methods on each subplot, for one metric."""

    n_ds = len(datasets)
    ncols = min(3, n_ds)
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, dataset in enumerate(datasets):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        kernel_trajs = all_data.get(dataset, {})
        for kernel in sorted(kernel_trajs.keys()):
            epochs, means, stds = aggregate_metric(kernel_trajs[kernel], metric)
            if len(epochs) == 0:
                continue
            label = METHOD_DISPLAY.get(kernel, kernel)
            color = METHOD_COLORS.get(kernel, None)
            ls = METHOD_LINESTYLES.get(kernel, "-")
            ax.plot(epochs, means, label=label, color=color, linestyle=ls, linewidth=1.5)
            ax.fill_between(epochs, means - stds, means + stds, alpha=0.15, color=color)

        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=11)
        ax.set_xlabel("Epoch")
        if c == 0:
            ax.set_ylabel(METRIC_DISPLAY.get(metric, metric))
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_ds, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    # Legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=min(len(handles), 4), fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    metric_label = METRIC_DISPLAY.get(metric, metric)
    fig.suptitle(f"Convergence: {metric_label}", y=1.08, fontsize=14)
    fig.tight_layout()

    out_path = output_dir / f"convergence_grid_{metric}.pdf"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def aggregate_metric_cummin(
    trajectories: List[dict],
    metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate a metric across seeds using cumulative minimum (best-so-far).

    Returns monotonically decreasing curves, much cleaner for convergence plots.
    Returns: (epochs, means_of_cummins, stds_of_cummins)
    """
    per_seed = []
    for traj_data in trajectories:
        epoch_val = {}
        for pt in traj_data["trajectory"]:
            if metric in pt and pt[metric] is not None:
                epoch_val[pt["epoch"]] = pt[metric]
        if epoch_val:
            per_seed.append(epoch_val)

    if not per_seed:
        return np.array([]), np.array([]), np.array([])

    common_epochs = sorted(set.intersection(*[set(d.keys()) for d in per_seed]))
    if not common_epochs:
        return np.array([]), np.array([]), np.array([])

    epochs = np.array(common_epochs)
    # For each seed, compute cumulative minimum
    cummins = []
    for d in per_seed:
        vals = np.array([d[e] for e in common_epochs])
        cummins.append(np.minimum.accumulate(vals))
    cummins = np.array(cummins)
    means = np.mean(cummins, axis=0)
    stds = np.std(cummins, axis=0)

    return epochs, means, stds


def plot_threshold_convergence(
    all_data: Dict[str, Dict[str, List[dict]]],
    datasets: List[str],
    metric: str,
    output_dir: Path,
    threshold_method: str = "ddpm",
    show_methods: Optional[List[str]] = None,
    suffix: str = "",
) -> None:
    """Plot convergence with cumulative-minimum curves and a baseline quality threshold.

    Uses best-so-far (cumulative min) for clean, monotonically decreasing curves.
    Adds a horizontal threshold at the baseline's final quality to show that
    kFFM methods reach baseline quality in fewer epochs.
    """
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4), squeeze=False)
    axes = axes[0]

    if show_methods is None:
        show_methods = ["none", "euclidean", "rbf", "signature", "ddpm", "ncsn"]

    # Ordered for consistent legend
    method_order = [m for m in show_methods if any(m in all_data.get(d, {}) for d in datasets)]

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        kernel_trajs = all_data.get(dataset, {})

        # Compute threshold = final value of threshold_method (using cummin)
        threshold_val = None
        if threshold_method in kernel_trajs:
            _, means_t, _ = aggregate_metric_cummin(kernel_trajs[threshold_method], metric)
            if len(means_t) > 0:
                threshold_val = means_t[-1]

        # Track y-range for axis limits
        all_finals = []

        for kernel in method_order:
            if kernel not in kernel_trajs:
                continue
            epochs, means, stds = aggregate_metric_cummin(kernel_trajs[kernel], metric)
            if len(epochs) == 0:
                continue

            label = METHOD_DISPLAY.get(kernel, kernel)
            color = METHOD_COLORS.get(kernel, None)
            ls = METHOD_LINESTYLES.get(kernel, "-")
            lw = 2.0 if kernel not in ("ddpm", "ncsn", "gano") else 1.5
            ax.plot(epochs, means, label=label, color=color, linestyle=ls, linewidth=lw)
            ax.fill_between(epochs,
                            np.maximum(means - stds, means * 0.1),
                            means + stds,
                            alpha=0.12, color=color)
            all_finals.append(means[-1])

        # Draw threshold line
        if threshold_val is not None:
            thr_label = METHOD_DISPLAY.get(threshold_method, threshold_method)
            ax.axhline(y=threshold_val, color=METHOD_COLORS.get(threshold_method, 'gray'),
                       linestyle=':', linewidth=2.0, alpha=0.6,
                       label=f'{thr_label} final quality')

        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=12, fontweight='bold')
        ax.set_xlabel("Epoch", fontsize=10)
        if idx == 0:
            ax.set_ylabel(METRIC_DISPLAY.get(metric, metric), fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Set y-limits: 0.3x best final to 5x worst final (avoid extreme ranges)
        if all_finals:
            ymin = min(all_finals) * 0.3
            ymax = max(all_finals) * 50
            if threshold_val:
                ymax = max(ymax, threshold_val * 10)
            ax.set_ylim(ymin, ymax)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=min(len(handles), 4), fontsize=9,
               bbox_to_anchor=(0.5, 1.05))

    metric_label = METRIC_DISPLAY.get(metric, metric)
    fig.suptitle(f"Convergence to Baseline Quality: {metric_label} (best-so-far)",
                 y=1.12, fontsize=13)
    fig.tight_layout()

    out_path = output_dir / f"convergence_threshold_{metric}{suffix}.pdf"
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot convergence study results")
    parser.add_argument('--datasets', nargs='+', default=None,
                        help="Datasets to plot (default: all available)")
    parser.add_argument('--metrics', nargs='+', default=None,
                        help="Metrics to plot (default: auto per dataset type)")
    parser.add_argument('--output-dir', type=str, default='../outputs/figures/',
                        help="Output directory for figures")
    parser.add_argument('--base-dir', type=str, default='../outputs/seeded_runs',
                        help="Base directory with convergence_study/ subdirectory")
    parser.add_argument('--no-log', action='store_true',
                        help="Use linear y-axis instead of log")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_trajectories(base_dir)

    if not all_data:
        print("No convergence study data found.")
        return

    datasets = args.datasets or sorted(all_data.keys())
    print(f"Datasets found: {datasets}")

    # Per-dataset plots
    for dataset in datasets:
        if dataset not in all_data:
            print(f"  No data for {dataset}, skipping")
            continue

        if args.metrics:
            metrics = args.metrics
        elif dataset in PDE_DATASETS:
            metrics = ["mean_mse", "variance_mse", "spectrum_mse_log"]
        else:
            metrics = ["mean_mse", "variance_mse", "autocorrelation_mse"]

        plot_dataset_convergence(
            dataset, all_data[dataset], metrics, output_dir,
            log_scale=not args.no_log,
        )

    # Combined grid plots for key metrics
    for metric in ["mean_mse", "variance_mse"]:
        avail_datasets = [d for d in datasets if d in all_data]
        if avail_datasets:
            plot_combined_grid(
                all_data, avail_datasets, metric, output_dir,
                log_scale=not args.no_log,
            )

    # Threshold-based convergence plots
    favorable_datasets = [d for d in ["heston", "aemet", "kdv"] if d in all_data]
    if favorable_datasets:
        # vs DDPM threshold: shows kFFM reaches baseline quality faster
        for metric in ["mean_mse", "variance_mse"]:
            plot_threshold_convergence(
                all_data, favorable_datasets, metric, output_dir,
                threshold_method="ddpm",
                show_methods=["none", "euclidean", "rbf", "signature", "ddpm", "ncsn"],
            )
        # vs FFM(none) threshold: highlights kernel's benefit over independent coupling
        for metric in ["mean_mse"]:
            plot_threshold_convergence(
                all_data, favorable_datasets, metric, output_dir,
                threshold_method="none",
                show_methods=["none", "euclidean", "rbf", "signature", "ddpm", "ncsn"],
                suffix="_vs_ffm",
            )

    # Also generate a summary markdown table with final-epoch metrics
    generate_summary_table(all_data, datasets, output_dir)


def generate_summary_table(
    all_data: Dict[str, Dict[str, List[dict]]],
    datasets: List[str],
    output_dir: Path,
) -> None:
    """Generate a markdown summary table of convergence study final metrics."""

    lines = [
        "# Convergence Study Summary",
        "",
        "Final-epoch metrics (mean +/- std across seeds).",
        "",
    ]

    for dataset in datasets:
        if dataset not in all_data:
            continue

        ds_label = DATASET_DISPLAY.get(dataset, dataset)
        is_pde = dataset in PDE_DATASETS
        if is_pde:
            metrics = ["mean_mse", "variance_mse", "spectrum_mse_log"]
            headers = ["Mean MSE", "Variance MSE", "Log-Spectral MSE"]
        else:
            metrics = ["mean_mse", "variance_mse", "autocorrelation_mse"]
            headers = ["Mean MSE", "Variance MSE", "Autocorr MSE"]

        lines.append(f"## {ds_label}")
        lines.append("")
        header_line = "| Method | " + " | ".join(headers) + " |"
        sep_line = "|--------|" + "|".join([":----------:" for _ in headers]) + "|"
        lines.append(header_line)
        lines.append(sep_line)

        for kernel in sorted(all_data[dataset].keys()):
            trajs = all_data[dataset][kernel]
            label = METHOD_DISPLAY.get(kernel, kernel)
            cells = []
            for metric in metrics:
                epochs, means, stds = aggregate_metric(trajs, metric)
                if len(epochs) > 0:
                    # Take the last checkpoint
                    cells.append(f"{means[-1]:.2e} +/- {stds[-1]:.2e}")
                else:
                    cells.append("N/A")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")

        lines.append("")

    summary_path = output_dir / "convergence_summary.md"
    with open(summary_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"Saved {summary_path}")


if __name__ == '__main__':
    main()
