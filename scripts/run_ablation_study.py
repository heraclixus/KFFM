"""
Hyperparameter ablation study for kFFM.

Sweeps one hyperparameter at a time while holding the others fixed,
then trains + evaluates to produce ablation curves.

Ablation axes:
  1. RBF bandwidth σ:    {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}   (fix ε=0.1)
  2. Sinkhorn ε:          {0.001, 0.01, 0.05, 0.1, 0.5, 1.0} (fix σ=best)
  3. Signature dyadic_order: {0, 1, 2, 3}                     (fix other sig params)
  4. Signature static_kernel_sigma: {0.1, 0.5, 1.0, 2.0, 5.0} (fix dyadic_order=1)

Each ablation point is a full train + evaluate cycle (same as run_seeded_experiments).
The independent (FFM) baseline is run alongside for comparison.

Usage:
    # Run one ablation axis for one dataset
    python run_ablation_study.py --dataset aemet --ablation rbf_sigma --seed 0

    # Run all axes for a dataset
    python run_ablation_study.py --dataset aemet --ablation all --seed 0
"""

import sys
sys.path.append('../')

import argparse
import json
import time
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, List, Tuple
from torch.utils.data import DataLoader

from run_seeded_experiments import (
    SETUP_FUNCTIONS,
    SPATIAL_2D_DATASETS,
    PDE_DATASETS,
    SEQUENCE_DATASETS,
    create_model_1d,
    create_model_2d,
    build_ffm_kwargs,
    get_seeds,
)


# ─────────────────────────────────────────────────────────────────────
# Ablation configurations
# ─────────────────────────────────────────────────────────────────────

# 1. RBF bandwidth σ sweep (fix ε = 0.1)
RBF_SIGMA_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

def rbf_sigma_configs() -> List[Tuple[str, dict]]:
    configs = []
    for sigma in RBF_SIGMA_VALUES:
        name = f"rbf_sigma{sigma}"
        cfg = {
            "use_ot": True,
            "ot_method": "sinkhorn",
            "ot_reg": 0.1,
            "ot_kernel": "rbf",
            "ot_coupling": "sample",
            "ot_kernel_params": {"sigma": sigma},
        }
        configs.append((name, cfg))
    return configs


# 2. Sinkhorn ε sweep (fix σ = 1.0 for RBF)
SINKHORN_EPS_VALUES = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

def sinkhorn_eps_configs() -> List[Tuple[str, dict]]:
    configs = []
    for eps in SINKHORN_EPS_VALUES:
        name = f"rbf_eps{eps}"
        cfg = {
            "use_ot": True,
            "ot_method": "sinkhorn",
            "ot_reg": eps,
            "ot_kernel": "rbf",
            "ot_coupling": "sample",
            "ot_kernel_params": {"sigma": 1.0},
        }
        configs.append((name, cfg))
    return configs


# 3. Signature dyadic order sweep
SIG_DYADIC_ORDERS = [0, 1, 2, 3]

def sig_dyadic_configs() -> List[Tuple[str, dict]]:
    configs = []
    for order in SIG_DYADIC_ORDERS:
        name = f"sig_order{order}"
        cfg = {
            "use_ot": True,
            "ot_method": "sinkhorn",
            "ot_reg": 0.1,
            "ot_kernel": "signature",
            "ot_coupling": "sample",
            "ot_kernel_params": {
                "time_aug": True,
                "lead_lag": False,
                "dyadic_order": order,
                "static_kernel_type": "rbf",
                "static_kernel_sigma": 1.0,
                "add_basepoint": True,
                "normalize": True,
                "max_seq_len": 64,
                "max_batch": 32,
            },
        }
        configs.append((name, cfg))
    return configs


# 4. Signature static kernel sigma sweep (fix dyadic_order = 1)
SIG_SIGMA_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]

def sig_sigma_configs() -> List[Tuple[str, dict]]:
    configs = []
    for sigma in SIG_SIGMA_VALUES:
        name = f"sig_sigma{sigma}"
        cfg = {
            "use_ot": True,
            "ot_method": "sinkhorn",
            "ot_reg": 0.1,
            "ot_kernel": "signature",
            "ot_coupling": "sample",
            "ot_kernel_params": {
                "time_aug": True,
                "lead_lag": False,
                "dyadic_order": 1,
                "static_kernel_type": "rbf",
                "static_kernel_sigma": sigma,
                "add_basepoint": True,
                "normalize": True,
                "max_seq_len": 64,
                "max_batch": 32,
            },
        }
        configs.append((name, cfg))
    return configs


# 5. Euclidean Sinkhorn ε sweep (no kernel params)
def euclidean_eps_configs() -> List[Tuple[str, dict]]:
    configs = []
    for eps in SINKHORN_EPS_VALUES:
        name = f"euc_eps{eps}"
        cfg = {
            "use_ot": True,
            "ot_method": "sinkhorn",
            "ot_reg": eps,
            "ot_kernel": "euclidean",
            "ot_coupling": "sample",
        }
        configs.append((name, cfg))
    return configs


# Independent baseline (always included)
INDEPENDENT_CONFIG = ("independent", {"use_ot": False})


ABLATION_AXES = {
    "rbf_sigma": rbf_sigma_configs,
    "sinkhorn_eps": sinkhorn_eps_configs,
    "sig_dyadic": sig_dyadic_configs,
    "sig_sigma": sig_sigma_configs,
    "euc_eps": euclidean_eps_configs,
}

# Which ablation axes apply to which dataset types
SEQUENCE_ABLATIONS = ["rbf_sigma", "sinkhorn_eps", "sig_dyadic", "sig_sigma", "euc_eps"]
PDE_ABLATIONS = ["rbf_sigma", "sinkhorn_eps", "euc_eps"]  # No signature for 2D PDEs
PDE_1D_ABLATIONS = ["rbf_sigma", "sinkhorn_eps", "sig_dyadic", "sig_sigma", "euc_eps"]  # KdV is 1D


# ─────────────────────────────────────────────────────────────────────
# Training + evaluation for one config
# ─────────────────────────────────────────────────────────────────────
def train_and_evaluate(
    config: dict,
    setup: dict,
    seed: int,
    device: str,
    train_data: torch.Tensor,
    ground_truth: torch.Tensor,
    save_dir: Path,
) -> Dict[str, Any]:
    """Train one FFM config and return quality metrics."""
    from functional_fm_ot import FFMModelOT
    from util.eval import GenerationQualityMetrics, GenerationQualityMetrics2D

    is_2d = setup.get("is_2d", False)

    # Create model
    if is_2d:
        model = create_model_2d(setup["modes"], setup["hch"], setup["pch"], device)
        sample_dims = list(setup["spatial_dims"])
    else:
        model = create_model_1d(setup["modes"], setup["width"], setup["mlp_width"], device)
        sample_dims = [setup["n_x"]]

    ffm_kwargs = build_ffm_kwargs(config, setup, device)
    ffm = FFMModelOT(model, **ffm_kwargs)

    # Use smaller batch for signature kernel
    kernel = config.get("ot_kernel", "")
    batch_size = setup["batch_size_sig"] if kernel == "signature" else setup["batch_size"]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)

    save_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    ffm.train(
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=setup["epochs"],
        scheduler=scheduler,
        eval_int=0,
        save_int=0,
        generate=False,
        save_path=save_dir,
    )
    train_time = time.time() - t_start

    # Generate samples
    n_gen = setup["n_gen_samples"]
    print(f"    Generating {n_gen} samples...")
    with torch.no_grad():
        model.eval()
        if is_2d:
            samples = ffm.sample(sample_dims, n_samples=n_gen).cpu().squeeze()
        else:
            samples = ffm.sample(sample_dims, n_samples=n_gen).cpu().squeeze()

    # Compute quality metrics
    if is_2d:
        qm = GenerationQualityMetrics2D(
            config_name="ablation", ot_kernel=kernel or "",
            ot_method=config.get("ot_method", ""), ot_coupling=config.get("ot_coupling", ""),
            use_ot=config.get("use_ot", False),
        )
    else:
        qm = GenerationQualityMetrics(
            config_name="ablation", ot_kernel=kernel or "",
            ot_method=config.get("ot_method", ""), ot_coupling=config.get("ot_coupling", ""),
            use_ot=config.get("use_ot", False),
        )
    qm.compute_from_samples(ground_truth, samples, compute_distributional=True)
    qm.total_train_time = train_time

    # Save
    torch.save(samples, save_dir / 'samples.pt')
    torch.save(model.state_dict(), save_dir / 'model.pt')
    qm.save(save_dir / 'quality_metrics.json')
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    return qm.to_dict()


# ─────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────
def run_ablation(
    dataset: str,
    ablation: str,
    seed_idx: int,
    skip_existing: bool = False,
) -> None:
    """Run one ablation axis for one dataset and one seed."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    seeds = get_seeds(10)
    seed = seeds[seed_idx]

    # Determine valid ablation axes for this dataset
    is_2d = dataset in SPATIAL_2D_DATASETS
    is_pde_1d = dataset in PDE_DATASETS and not is_2d  # kdv, stochastic_kdv

    if ablation == "all":
        if is_2d:
            axes = PDE_ABLATIONS
        elif is_pde_1d:
            axes = PDE_1D_ABLATIONS
        else:
            axes = SEQUENCE_ABLATIONS
    else:
        # Validate
        if ablation.startswith("sig_") and is_2d:
            print(f"Signature ablations not supported for 2D dataset {dataset}")
            return
        axes = [ablation]

    # Setup dataset
    setup_fn = SETUP_FUNCTIONS[dataset]
    setup = setup_fn()

    is_multi = setup.get("is_multi", False)
    if is_multi:
        first_sub = list(setup["datasets"].keys())[0]
        train_data = setup["datasets"][first_sub]["train_data"]
        ground_truth = setup["datasets"][first_sub]["ground_truth"]
    else:
        train_data = setup["train_data"]
        ground_truth = setup["ground_truth"]

    output_base = Path('../outputs/seeded_runs/ablation_study') / dataset

    for axis in axes:
        print(f"\n{'=' * 70}")
        print(f"Ablation: {axis} | Dataset: {dataset} | Seed: {seed}")
        print(f"{'=' * 70}")

        config_fn = ABLATION_AXES[axis]
        configs = config_fn()

        # Always include independent baseline
        all_configs = [INDEPENDENT_CONFIG] + configs

        axis_results = []

        for config_name, config in all_configs:
            save_dir = output_base / axis / config_name / f"seed_{seed}"
            metrics_path = save_dir / 'quality_metrics.json'

            if skip_existing and metrics_path.exists():
                print(f"\n  Skipping {config_name} (exists)")
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                axis_results.append({"config_name": config_name, "config": config, "metrics": metrics})
                continue

            print(f"\n  Running {config_name}...")

            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            metrics = train_and_evaluate(
                config, setup, seed, device,
                train_data, ground_truth, save_dir,
            )
            axis_results.append({"config_name": config_name, "config": config, "metrics": metrics})

        # Save axis summary
        summary_dir = output_base / axis
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"ablation_results_seed{seed}.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "dataset": dataset,
                "ablation_axis": axis,
                "seed": seed,
                "seed_idx": seed_idx,
                "results": axis_results,
            }, f, indent=2, default=str)
        print(f"\n  Saved {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter ablation study")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(SETUP_FUNCTIONS.keys()),
                        help="Dataset name")
    parser.add_argument('--ablation', type=str, required=True,
                        choices=list(ABLATION_AXES.keys()) + ['all'],
                        help="Ablation axis to sweep")
    parser.add_argument('--seed', type=int, required=True,
                        help="Seed index (0-9)")
    parser.add_argument('--skip-existing', action='store_true',
                        help="Skip configs with existing quality_metrics.json")
    args = parser.parse_args()

    run_ablation(
        dataset=args.dataset,
        ablation=args.ablation,
        seed_idx=args.seed,
        skip_existing=args.skip_existing,
    )


if __name__ == '__main__':
    main()
