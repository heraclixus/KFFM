#!/usr/bin/env python3
"""Create exact-path PDE sample figures for the updated RBF results.

The older generic sample plotting script picks the first matching RBF directory,
which can silently select stale runs. This script is intentionally explicit:
it points at the current promoted NS and Stoch. NS configs and compares them
against the relevant FFM / L2 / older RBF references.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from util.util import load_navier_stokes, load_stochastic_ns  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DATASETS = {
    "navier_stokes": {
        "title": "Navier-Stokes",
        "loader": lambda: load_navier_stokes(
            PROJECT_ROOT / "data/ns.mat", shuffle=False, subsample_time=5
        ).squeeze(1),
        "root": PROJECT_ROOT / "outputs/seeded_runs/navier_stokes",
        "methods": [
            ("Data", None),
            ("FFM", "cfm_indep"),
            ("FFM-L2", "cfm_ot"),
            ("RBF (old)", "cfm_rbf_ot_r001"),
            ("RBF (new)", "ns_gp005_rbf_s30_r005"),
        ],
    },
    "stochastic_ns": {
        "title": "Stoch. NS",
        "loader": lambda: load_stochastic_ns(
            PROJECT_ROOT / "data/stochastic_ns_64.mat", shuffle=False, subsample_time=5
        ).squeeze(1),
        "root": PROJECT_ROOT / "outputs/seeded_runs/stochastic_ns",
        "methods": [
            ("Data", None),
            ("FFM", "cfm_indep"),
            ("FFM-L2", "cfm_ot"),
            ("RBF (old)", "cfm_rbf_ot_s05"),
            ("RBF (new)", "sns_iid_rbf_s025_r005"),
        ],
    },
}


def load_samples(root: Path, method: str, seed: int) -> torch.Tensor:
    path = root / method / f"seed_{seed}" / "samples.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        for key in ("samples", "data", "x", "X"):
            if key in data and isinstance(data[key], torch.Tensor):
                data = data[key]
                break
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(data)!r}")
    return data.squeeze()


def field_std(samples: torch.Tensor) -> torch.Tensor:
    x = samples.float()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    return x.flatten(1).std(dim=1)


def choose_representative(samples: torch.Tensor, target_std: float) -> int:
    stats = field_std(samples)
    return int(torch.argmin(torch.abs(stats - target_std)).item())


def choose_quantile_samples(samples: torch.Tensor, quantiles: tuple[float, ...]) -> list[int]:
    stats = field_std(samples)
    if len(stats) == 0:
        return []
    order = torch.argsort(stats)
    indices: list[int] = []
    for q in quantiles:
        pos = int(round(q * (len(order) - 1)))
        idx = int(order[pos].item())
        if idx not in indices:
            indices.append(idx)
    while len(indices) < len(quantiles):
        candidate = int(order[min(len(indices), len(order) - 1)].item())
        if candidate not in indices:
            indices.append(candidate)
        else:
            break
    return indices


def tensor_to_image(sample: torch.Tensor) -> np.ndarray:
    arr = sample.detach().cpu().float().squeeze().numpy()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D field after squeeze, got shape {arr.shape}")
    return arr


def load_dataset_panel(
    dataset_key: str,
    seed: int,
    max_gt_for_median: int = 5000,
) -> tuple[dict, list[tuple[str, np.ndarray, int]], float]:
    spec = DATASETS[dataset_key]
    gt = spec["loader"]()
    gt = gt[:max_gt_for_median]
    target_std = float(torch.median(field_std(gt)).item())

    panels: list[tuple[str, np.ndarray, int]] = []
    gt_idx = choose_representative(gt, target_std)
    panels.append(("Data", tensor_to_image(gt[gt_idx]), gt_idx))

    for label, method in spec["methods"][1:]:
        samples = load_samples(spec["root"], method, seed)
        idx = choose_representative(samples, target_std)
        panels.append((label, tensor_to_image(samples[idx]), idx))

    all_values = np.concatenate([p[1].ravel() for p in panels])
    vmax = float(np.quantile(np.abs(all_values), 0.995))
    vmax = max(vmax, 1e-6)
    return spec, panels, vmax


def plot_combined(seed: int, output: Path) -> None:
    dataset_keys = ["navier_stokes", "stochastic_ns"]
    loaded = [load_dataset_panel(key, seed) for key in dataset_keys]

    fig, axes = plt.subplots(
        len(dataset_keys),
        5,
        figsize=(10.8, 4.55),
        constrained_layout=True,
    )

    for row, (spec, panels, vmax) in enumerate(loaded):
        for col, (label, image, _idx) in enumerate(panels):
            ax = axes[row, col]
            im = ax.imshow(
                image,
                origin="lower",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=11, fontweight="bold", pad=5)
            if col == 0:
                ax.set_ylabel(spec["title"], fontsize=11, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)
                spine.set_color("#444444")

        cbar = fig.colorbar(
            im,
            ax=axes[row, :],
            fraction=0.014,
            pad=0.012,
            aspect=18,
        )
        cbar.ax.tick_params(labelsize=7, width=0.4, length=2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Saved {output.with_suffix('.png')}")


def plot_montage(dataset_key: str, seed: int, output: Path) -> None:
    spec = DATASETS[dataset_key]
    quantiles = (0.18, 0.38, 0.62, 0.82)
    method_rows = spec["methods"]

    tensors: list[tuple[str, torch.Tensor]] = [("Data", spec["loader"]()[:5000])]
    for label, method in method_rows[1:]:
        tensors.append((label, load_samples(spec["root"], method, seed)))

    selected: list[tuple[str, list[np.ndarray]]] = []
    all_images: list[np.ndarray] = []
    for label, tensor in tensors:
        idxs = choose_quantile_samples(tensor, quantiles)
        images = [tensor_to_image(tensor[idx]) for idx in idxs]
        selected.append((label, images))
        all_images.extend(images)

    all_values = np.concatenate([x.ravel() for x in all_images])
    vmax = float(np.quantile(np.abs(all_values), 0.995))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(
        len(selected),
        len(quantiles),
        figsize=(7.2, 8.0),
        constrained_layout=True,
    )

    for row, (label, images) in enumerate(selected):
        for col, image in enumerate(images):
            ax = axes[row, col]
            im = ax.imshow(
                image,
                origin="lower",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"q{quantiles[col]:.2f}", fontsize=9, pad=3)
            if col == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_linewidth(0.35)
                spine.set_color("#444444")

    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.012, aspect=24)
    cbar.ax.tick_params(labelsize=7, width=0.4, length=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Saved {output.with_suffix('.png')}")


def plot_single(dataset_key: str, seed: int, output: Path) -> None:
    spec, panels, vmax = load_dataset_panel(dataset_key, seed)
    fig, axes = plt.subplots(1, 5, figsize=(10.8, 2.1), constrained_layout=True)
    for ax, (label, image, _idx) in zip(axes, panels):
        im = ax.imshow(
            image,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(label, fontsize=11, fontweight="bold", pad=5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#444444")
    cbar = fig.colorbar(im, ax=axes, fraction=0.014, pad=0.012, aspect=18)
    cbar.ax.tick_params(labelsize=7, width=0.4, length=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Saved {output.with_suffix('.png')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/figures",
    )
    args = parser.parse_args()

    plot_combined(args.seed, args.output_dir / "pde_best_samples_updated.pdf")
    plot_single(
        "navier_stokes",
        args.seed,
        args.output_dir / "navier_stokes_sample_comparison_updated.pdf",
    )
    plot_single(
        "stochastic_ns",
        args.seed,
        args.output_dir / "stochastic_ns_sample_comparison_updated.pdf",
    )
    plot_montage(
        "navier_stokes",
        args.seed,
        args.output_dir / "navier_stokes_sample_montage_updated.pdf",
    )
    plot_montage(
        "stochastic_ns",
        args.seed,
        args.output_dir / "stochastic_ns_sample_montage_updated.pdf",
    )


if __name__ == "__main__":
    main()
