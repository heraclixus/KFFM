"""
Compute distributional metrics (MMD, Sliced Wasserstein, Marginal Wasserstein)
from existing samples.pt files without retraining.

Loads saved samples and ground truth, computes distributional metrics,
and merges them into existing quality_metrics.json files.

Usage:
    # Process a specific dataset/kernel/seed
    python compute_distributional_metrics.py --dataset kdv --kernel euclidean --seed 0

    # Process all seeds for a dataset/kernel
    python compute_distributional_metrics.py --dataset kdv --kernel euclidean

    # Process everything
    python compute_distributional_metrics.py --all
"""

import sys
sys.path.append('../')

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from util.eval import compute_mmd_rbf, compute_sliced_wasserstein, compute_marginal_wasserstein

# Import dataset setup functions from run_seeded_experiments (same directory)
from run_seeded_experiments import (
    setup_kdv, setup_navier_stokes, setup_stochastic_kdv, setup_stochastic_ns,
    setup_aemet, setup_expr_genes, setup_economy,
    setup_heston,
    setup_heston_long,
    SPATIAL_2D_DATASETS,
)

DATASET_SETUP = {
    "kdv": setup_kdv,
    "navier_stokes": setup_navier_stokes,
    "stochastic_kdv": setup_stochastic_kdv,
    "stochastic_ns": setup_stochastic_ns,
    "aemet": setup_aemet,
    "expr_genes": setup_expr_genes,
    "economy": setup_economy,
    "heston": setup_heston,
    "heston-long": setup_heston_long,
}

ALL_KERNELS = ["none", "euclidean", "rbf", "signature", "gaussian", "ddpm", "ncsn", "gano"]


def _to_python_float(val):
    """Convert numpy/torch floats to Python float for JSON serialization."""
    if val is None:
        return None
    if hasattr(val, 'item'):
        return float(val.item())
    if isinstance(val, (np.floating, np.integer)):
        return float(val)
    return float(val)


def compute_distributional_for_seed(
    seed_dir: Path,
    ground_truth: torch.Tensor,
    n_projections: int = 100,
):
    """Compute distributional metrics for a single seed directory.

    Loads samples.pt, computes MMD/SW/MW, and merges into quality_metrics.json.

    Returns dict of new metrics or None if samples.pt not found.
    """
    samples_path = seed_dir / 'samples.pt'
    if not samples_path.exists():
        print(f"  No samples.pt in {seed_dir}, skipping.")
        return None

    metrics_path = seed_dir / 'quality_metrics.json'

    # Load samples
    samples = torch.load(samples_path, weights_only=False)
    if isinstance(samples, dict):
        # Some datasets save as dict
        samples = samples.get('samples', samples.get('generated', None))
        if samples is None:
            print(f"  Cannot extract samples from dict in {seed_dir}")
            return None

    # Ensure CPU tensors
    if samples.is_cuda:
        samples = samples.cpu()
    gt = ground_truth.cpu() if ground_truth.is_cuda else ground_truth

    # Match sample counts: use the minimum of both
    n_gen = samples.shape[0]
    n_real = gt.shape[0]
    n_use = min(n_gen, n_real)
    samples_use = samples[:n_use]
    gt_use = gt[:n_use]

    print(f"  Computing distributional metrics (n={n_use}, shape={samples_use.shape[1:]})...")

    new_metrics = {}

    # MMD-RBF
    try:
        mmd = compute_mmd_rbf(gt_use, samples_use)
        new_metrics['mmd_rbf'] = _to_python_float(mmd)
        print(f"    MMD-RBF: {mmd:.6e}")
    except Exception as e:
        print(f"    MMD-RBF failed: {e}")
        new_metrics['mmd_rbf'] = None

    # Sliced Wasserstein
    try:
        sw = compute_sliced_wasserstein(gt_use, samples_use, n_projections=n_projections)
        new_metrics['sliced_wasserstein'] = _to_python_float(sw)
        print(f"    Sliced Wasserstein: {sw:.6e}")
    except Exception as e:
        print(f"    Sliced Wasserstein failed: {e}")
        new_metrics['sliced_wasserstein'] = None

    # Marginal Wasserstein
    try:
        mw = compute_marginal_wasserstein(gt_use, samples_use)
        new_metrics['marginal_wasserstein'] = _to_python_float(mw)
        print(f"    Marginal Wasserstein: {mw:.6e}")
    except Exception as e:
        print(f"    Marginal Wasserstein failed: {e}")
        new_metrics['marginal_wasserstein'] = None

    # Merge into existing quality_metrics.json
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            existing = json.load(f)
    else:
        existing = {}

    existing.update(new_metrics)

    with open(metrics_path, 'w') as f:
        json.dump(existing, f, indent=2)

    print(f"  Updated {metrics_path}")
    return new_metrics


def get_ground_truth(dataset: str):
    """Load ground truth for a dataset."""
    if dataset not in DATASET_SETUP:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_SETUP.keys())}")

    print(f"Loading dataset '{dataset}'...")
    setup = DATASET_SETUP[dataset]()

    if setup.get("is_multi", False):
        # Multi-dataset (e.g. economy): return dict of ground truths
        return {name: sub["ground_truth"] for name, sub in setup["datasets"].items()}
    else:
        return setup["ground_truth"]


def process_dataset_kernel(dataset: str, kernel: str, base_dir: Path, seed: int = None):
    """Process all seeds (or a specific seed) for a dataset/kernel."""
    kernel_dir = base_dir / dataset / kernel
    if not kernel_dir.exists():
        print(f"Directory not found: {kernel_dir}, skipping.")
        return

    # Load ground truth once
    gt_data = get_ground_truth(dataset)

    if seed is not None:
        seeds = [seed]
    else:
        # Find all seed directories
        seed_dirs = sorted(kernel_dir.glob("seed_*"))
        seeds = [int(d.name.split('_')[1]) for d in seed_dirs]

    if not seeds:
        print(f"No seed directories found in {kernel_dir}")
        return

    print(f"\nProcessing {dataset}/{kernel} ({len(seeds)} seeds)...")

    for s in seeds:
        seed_dir = kernel_dir / f"seed_{s}"
        if not seed_dir.exists():
            # Check for sub-dataset directories (economy has econ1_population, etc.)
            if isinstance(gt_data, dict):
                for sub_name, sub_gt in gt_data.items():
                    sub_seed_dir = kernel_dir / sub_name / f"seed_{s}"
                    if sub_seed_dir.exists():
                        print(f"\n  Seed {s} / {sub_name}:")
                        compute_distributional_for_seed(sub_seed_dir, sub_gt)
            else:
                print(f"  seed_{s} not found, skipping.")
            continue

        if isinstance(gt_data, dict):
            # Multi-dataset: check sub-directories
            for sub_name, sub_gt in gt_data.items():
                sub_seed_dir = kernel_dir / sub_name / f"seed_{s}"
                if sub_seed_dir.exists():
                    print(f"\n  Seed {s} / {sub_name}:")
                    compute_distributional_for_seed(sub_seed_dir, sub_gt)
        else:
            print(f"\n  Seed {s}:")
            compute_distributional_for_seed(seed_dir, gt_data)


def main():
    parser = argparse.ArgumentParser(description="Compute distributional metrics from saved samples")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--kernel", type=str, help="Kernel name")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed (default: all)")
    parser.add_argument("--all", action="store_true", help="Process all dataset/kernel combos")
    parser.add_argument("--input_dir", type=str, default="../outputs/seeded_runs",
                        help="Base directory for seeded runs")
    parser.add_argument("--n_projections", type=int, default=100,
                        help="Number of projections for sliced Wasserstein")
    args = parser.parse_args()

    base_dir = Path(args.input_dir)

    if args.all:
        # Discover all dataset/kernel combos
        if not base_dir.exists():
            print(f"Base directory not found: {base_dir}")
            return
        for dataset_dir in sorted(base_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = dataset_dir.name
            for kernel_dir in sorted(dataset_dir.iterdir()):
                if not kernel_dir.is_dir():
                    continue
                kernel = kernel_dir.name
                try:
                    process_dataset_kernel(dataset, kernel, base_dir, seed=args.seed)
                except Exception as e:
                    print(f"ERROR processing {dataset}/{kernel}: {e}")
                    import traceback
                    traceback.print_exc()
    elif args.dataset and args.kernel:
        process_dataset_kernel(args.dataset, args.kernel, base_dir, seed=args.seed)
    else:
        parser.print_help()
        print("\nError: specify --dataset and --kernel, or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
