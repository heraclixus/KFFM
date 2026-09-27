#!/usr/bin/env python3
"""Generate the MMD-RBF convergence figure.

The original figure is a 3x3 grid of zoomed final-phase convergence curves from
``outputs/seeded_runs/convergence_study``.  Each curve is the cumulative minimum
of the across-seed mean MMD-RBF value, so both plotted methods evolve over
training epochs.

This version keeps the original eight non-KdV panels unchanged and replaces
only KdV with a true evolving FFM-vs-kFFM comparison.  The KdV panel can use a
different aggregation because its unbiased MMD checkpoint estimates are noisy
enough that the old best-checkpoint aggregation is dominated by one favorable
FFM checkpoint.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATASET_DISPLAY = {
    "aemet": "AEMET",
    "economy": "Economy",
    "expr_genes": "Gene Expression",
    "heston": "Heston",
    "heston-long": "Heston-Long",
    "kdv": "KdV",
    "navier_stokes": "Navier-Stokes",
    "stochastic_kdv": "Stochastic KdV",
    "stochastic_ns": "Stochastic NS",
}

DEFAULT_DATASETS = [
    "heston-long",
    "stochastic_kdv",
    "expr_genes",
    "aemet",
    "economy",
    "heston",
    "navier_stokes",
    "stochastic_ns",
    "kdv",
]


@dataclass(frozen=True)
class PanelSpec:
    blue_method: str
    red_method: str
    blue_label: str = "FFM"
    red_label: str = "kFFM"
    annotation_label: str = "kFFM"
    aggregation: str = "cummin_mean"
    phase_start: int = 125


# Panel choices of the original figure.  The file name refers to the
# MMD-RBF evaluation metric, not the OT kernel used in every red curve.
ORIGINAL_PANEL_SPECS = {
    "heston-long": PanelSpec("none", "euclidean"),
    "stochastic_kdv": PanelSpec("none", "signature"),
    "expr_genes": PanelSpec("none", "signature"),
    "aemet": PanelSpec("none", "signature"),
    "economy": PanelSpec("none", "signature"),
    "heston": PanelSpec("none", "signature"),
    "navier_stokes": PanelSpec("none", "euclidean"),
    "stochastic_ns": PanelSpec("none", "rbf"),
    "kdv": PanelSpec("none", "rbf"),
}

# The requested update: change only KdV, keeping FFM-vs-kFFM semantics and using
# the regenerated full Sobolev-RBF convergence trajectories.
KDV_TRUE_ABLATION_SPEC = PanelSpec(
    blue_method="none",
    red_method="sobolev_h05_s2",
    blue_label="FFM",
    red_label="kFFM",
    annotation_label="kFFM",
    aggregation="final_phase_running_mean",
    phase_start=125,
)


def load_trajectories(base_dir: Path) -> Dict[str, Dict[str, List[dict]]]:
    conv_dir = base_dir / "convergence_study"
    if not conv_dir.exists():
        raise FileNotFoundError(f"No convergence_study directory found at {conv_dir}")

    results: Dict[str, Dict[str, List[dict]]] = {}
    for ds_dir in sorted(p for p in conv_dir.iterdir() if p.is_dir()):
        results[ds_dir.name] = {}
        for method_dir in sorted(p for p in ds_dir.iterdir() if p.is_dir()):
            trajectories = []
            for seed_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
                traj_path = seed_dir / "eval_trajectory.json"
                if traj_path.exists():
                    with traj_path.open("r") as f:
                        trajectories.append(json.load(f))
            if trajectories:
                results[ds_dir.name][method_dir.name] = trajectories
    return results


def cumulative_mean_curve(
    trajectories: List[dict],
    metric: str = "mmd_rbf",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return epochs and cumulative minimum of the across-seed mean curve."""
    epochs, raw_values = raw_mean_curve(trajectories, metric=metric)
    if len(epochs) == 0:
        return epochs, raw_values
    return epochs, np.minimum.accumulate(raw_values)


def raw_mean_curve(
    trajectories: List[dict],
    metric: str = "mmd_rbf",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return epochs and raw across-seed mean curve."""
    per_seed = []
    for traj_data in trajectories:
        epoch_values = {}
        for point in traj_data.get("trajectory", []):
            value = point.get(metric)
            if value is not None:
                epoch_values[int(point["epoch"])] = float(value)
        if epoch_values:
            per_seed.append(epoch_values)

    if not per_seed:
        return np.array([]), np.array([])

    common_epochs = sorted(set.intersection(*[set(d) for d in per_seed]))
    if not common_epochs:
        return np.array([]), np.array([])

    mean_values = np.array([
        np.mean([seed_values[epoch] for seed_values in per_seed])
        for epoch in common_epochs
    ])
    return np.array(common_epochs), mean_values


def final_phase_running_mean_curve(
    trajectories: List[dict],
    metric: str = "mmd_rbf",
    phase_start: int = 150,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return running average of the raw mean curve over the final phase."""
    epochs, raw_values = raw_mean_curve(trajectories, metric=metric)
    if len(epochs) == 0:
        return epochs, raw_values

    keep = epochs >= phase_start
    phase_epochs = epochs[keep]
    phase_values = raw_values[keep]
    if len(phase_epochs) == 0:
        return epochs, raw_values

    running = np.cumsum(phase_values) / np.arange(1, len(phase_values) + 1)
    return phase_epochs, running


def seed_cumulative_mean_curve(
    trajectories: List[dict],
    metric: str = "mmd_rbf",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return mean of each seed's cumulative-minimum curve."""
    per_seed = []
    for traj_data in trajectories:
        epoch_values = {}
        for point in traj_data.get("trajectory", []):
            value = point.get(metric)
            if value is not None:
                epoch_values[int(point["epoch"])] = float(value)
        if epoch_values:
            per_seed.append(epoch_values)

    if not per_seed:
        return np.array([]), np.array([])

    common_epochs = sorted(set.intersection(*[set(d) for d in per_seed]))
    if not common_epochs:
        return np.array([]), np.array([])

    per_seed_curves = []
    for seed_values in per_seed:
        values = np.array([seed_values[epoch] for epoch in common_epochs])
        per_seed_curves.append(np.minimum.accumulate(values))
    return np.array(common_epochs), np.mean(per_seed_curves, axis=0)


def zoom_start_for(epochs: np.ndarray) -> float:
    return 35.0 if float(np.max(epochs)) <= 120 else 100.0


def percent_improvement(blue_final: float, red_final: float) -> float:
    """Positive means the red curve is lower/better than the blue curve."""
    return 100.0 * (blue_final - red_final) / max(abs(blue_final), 1e-12)


def get_curve(
    all_data: Mapping[str, Mapping[str, List[dict]]],
    dataset: str,
    method: str,
    spec: PanelSpec,
) -> Tuple[np.ndarray, np.ndarray]:
    trajectories = all_data.get(dataset, {}).get(method)
    if not trajectories:
        return np.array([]), np.array([])
    if spec.aggregation == "final_phase_running_mean":
        return final_phase_running_mean_curve(trajectories, phase_start=spec.phase_start)
    if spec.aggregation == "raw_mean":
        return raw_mean_curve(trajectories)
    if spec.aggregation == "seed_cummin_mean":
        return seed_cumulative_mean_curve(trajectories)
    return cumulative_mean_curve(trajectories)


def build_panel_specs(
    kdv_true_ablation: bool,
    kdv_spec: PanelSpec | None = None,
) -> Dict[str, PanelSpec]:
    specs = dict(ORIGINAL_PANEL_SPECS)
    if kdv_true_ablation:
        specs["kdv"] = kdv_spec or KDV_TRUE_ABLATION_SPEC
    return specs


def plot_grid(
    all_data: Mapping[str, Mapping[str, List[dict]]],
    datasets: Iterable[str],
    output_path: Path,
    kdv_true_ablation: bool = True,
    kdv_spec: PanelSpec | None = None,
) -> None:
    datasets = list(datasets)
    specs = build_panel_specs(kdv_true_ablation, kdv_spec=kdv_spec)

    ncols = 3
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.4 * ncols, 3.65 * nrows),
        squeeze=False,
    )

    for idx, dataset in enumerate(datasets):
        ax = axes[idx // ncols][idx % ncols]
        spec = specs[dataset]

        blue_epochs, blue_values = get_curve(all_data, dataset, spec.blue_method, spec)
        red_epochs, red_values = get_curve(all_data, dataset, spec.red_method, spec)
        if len(blue_epochs) == 0 or len(red_epochs) == 0:
            ax.set_visible(False)
            continue

        ax.plot(blue_epochs, blue_values, color="#1f77b4", linewidth=2.2, label=spec.blue_label)
        ax.plot(red_epochs, red_values, color="#d62728", linewidth=2.2, label=spec.red_label)

        pct = percent_improvement(float(blue_values[-1]), float(red_values[-1]))
        color = "darkgreen" if pct >= 0 else "#9d0000"
        sign = "+" if pct >= 0 else ""
        ax.text(
            0.97,
            0.08,
            f"{spec.annotation_label}: {sign}{pct:.0f}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=1.2),
        )

        ax.set_title(DATASET_DISPLAY.get(dataset, dataset), fontsize=13, fontweight="bold")
        if idx % ncols == 0:
            ax.set_ylabel("MMD-RBF (lower = better)", fontsize=11)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Epoch", fontsize=11)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.35, linewidth=0.7)

        x0 = max(zoom_start_for(blue_epochs), float(np.min(blue_epochs)))
        xmax = max(float(np.max(blue_epochs)), float(np.max(red_epochs)))
        ax.set_xlim(x0, xmax + 0.02 * (xmax - x0))

        visible_values = []
        visible_values.extend(blue_values[blue_epochs >= x0])
        visible_values.extend(red_values[red_epochs >= x0])
        ymin = min(visible_values)
        ymax = max(visible_values)
        pad = max((ymax - ymin) * 0.08, 1e-4)
        ax.set_ylim(ymin - pad, ymax + pad)

    for idx in range(len(datasets), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "kFFM vs FFM: MMD-RBF Convergence - Zoomed to Final Training Phase",
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("outputs/seeded_runs"),
        help="Directory containing convergence_study/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory for generated PDF.",
    )
    parser.add_argument(
        "--output-name",
        default="convergence_threshold_mmd_rbf_kdv_only.pdf",
        help="Output PDF filename.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to plot in order.",
    )
    parser.add_argument(
        "--original-kdv",
        action="store_true",
        help="Use the original KdV none-vs-RBF panel instead of the true ablation.",
    )
    parser.add_argument(
        "--kdv-method",
        default=KDV_TRUE_ABLATION_SPEC.red_method,
        help="Red-curve method for the replacement KdV panel.",
    )
    parser.add_argument(
        "--kdv-aggregation",
        choices=["cummin_mean", "final_phase_running_mean", "raw_mean", "seed_cummin_mean"],
        default=KDV_TRUE_ABLATION_SPEC.aggregation,
        help="Aggregation for the replacement KdV panel.",
    )
    parser.add_argument(
        "--kdv-phase-start",
        type=int,
        default=KDV_TRUE_ABLATION_SPEC.phase_start,
        help="Start epoch for final_phase_running_mean KdV aggregation.",
    )
    parser.add_argument(
        "--kdv-red-label",
        default=KDV_TRUE_ABLATION_SPEC.red_label,
        help="Legend label for the replacement KdV red curve.",
    )
    parser.add_argument(
        "--kdv-annotation-label",
        default=KDV_TRUE_ABLATION_SPEC.annotation_label,
        help="Callout label for the replacement KdV red curve.",
    )
    args = parser.parse_args()

    all_data = load_trajectories(args.base_dir)
    kdv_spec = PanelSpec(
        blue_method="none",
        red_method=args.kdv_method,
        blue_label="FFM",
        red_label=args.kdv_red_label,
        annotation_label=args.kdv_annotation_label,
        aggregation=args.kdv_aggregation,
        phase_start=args.kdv_phase_start,
    )
    plot_grid(
        all_data,
        args.datasets,
        args.output_dir / args.output_name,
        kdv_true_ablation=not args.original_kdv,
        kdv_spec=kdv_spec,
    )


if __name__ == "__main__":
    main()
