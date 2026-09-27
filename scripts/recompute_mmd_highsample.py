"""
Recompute MMD-RBF with more generated samples for tied datasets.

Loads saved model.pt, generates N samples (default 2000), and recomputes
MMD-RBF. Saves result to quality_metrics_highsample.json alongside original.

Usage:
    python recompute_mmd_highsample.py --dataset heston --method cfm_sig_ot --n-samples 2000
    python recompute_mmd_highsample.py --dataset navier_stokes --method cfm_rbf_ot --n-samples 1000
"""

import sys
sys.path.append('../')

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from functional_fm_ot import FFMModelOT
from util.eval import compute_mmd_rbf, compute_sliced_wasserstein

# Import setup functions and model creators from run_seeded_experiments
from run_seeded_experiments import (
    SETUP_FUNCTIONS, create_model_1d, create_model_2d, build_ffm_kwargs,
    CFM_OT_CONFIG, CFM_INDEP_CONFIG, CFM_RBF_OT_CONFIG, CFM_SIG_OT_CONFIG,
    CFM_RBF_OT_SIGMA_OVERRIDES, CFM_RBF_OT_SIGMA_VARIANTS, CFM_RBF_OT_REG_VARIANTS,
    CFM_RBF_OT_JOINT_VARIANTS, FALLBACK_CONFIGS, DIFFUSION_CONFIGS,
)

def get_config(method, dataset):
    """Get config for a method."""
    if method == 'cfm_ot':
        return CFM_OT_CONFIG.copy()
    elif method == 'cfm_indep':
        return CFM_INDEP_CONFIG.copy()
    elif method == 'cfm_rbf_ot':
        config = CFM_RBF_OT_CONFIG.copy()
        if dataset in CFM_RBF_OT_SIGMA_OVERRIDES:
            config["ot_kernel_params"] = {"sigma": CFM_RBF_OT_SIGMA_OVERRIDES[dataset]}
        return config
    elif method == 'cfm_sig_ot':
        return CFM_SIG_OT_CONFIG.copy()
    elif method in CFM_RBF_OT_SIGMA_VARIANTS:
        config = CFM_RBF_OT_CONFIG.copy()
        config["ot_kernel_params"] = {"sigma": CFM_RBF_OT_SIGMA_VARIANTS[method]}
        return config
    elif method in CFM_RBF_OT_REG_VARIANTS:
        config = CFM_RBF_OT_CONFIG.copy()
        config["ot_reg"] = CFM_RBF_OT_REG_VARIANTS[method]
        if dataset in CFM_RBF_OT_SIGMA_OVERRIDES:
            config["ot_kernel_params"] = {"sigma": CFM_RBF_OT_SIGMA_OVERRIDES[dataset]}
        return config
    elif method in CFM_RBF_OT_JOINT_VARIANTS:
        sigma, reg = CFM_RBF_OT_JOINT_VARIANTS[method]
        config = CFM_RBF_OT_CONFIG.copy()
        config["ot_reg"] = reg
        config["ot_kernel_params"] = {"sigma": sigma}
        return config
    elif method in ('none', 'euclidean', 'rbf', 'signature'):
        return FALLBACK_CONFIGS.get(method, FALLBACK_CONFIGS['none']).copy()
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--seed-idx', type=int, default=None,
                        help='Single seed index (0-9). If not set, runs all seeds.')
    parser.add_argument('--base-dir', default='../outputs/seeded_runs')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = Path(args.base_dir)
    dataset = args.dataset
    method = args.method
    n_samples = args.n_samples

    # Setup dataset
    setup_fn = SETUP_FUNCTIONS[dataset]
    setup = setup_fn()
    is_2d = setup.get("is_2d", False)
    ground_truth = setup["ground_truth"]

    # Get config
    config = get_config(method, dataset)

    # Find seed directories
    method_dir = base_dir / dataset / method
    if not method_dir.exists():
        print(f"No directory: {method_dir}")
        return

    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    seed_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    if args.seed_idx is not None:
        seeds = [args.seed_idx]

    for seed_idx in seeds:
        seed = seed_values[seed_idx]
        seed_dir = method_dir / f"seed_{seed}"
        model_path = seed_dir / "model.pt"

        if not model_path.exists():
            print(f"  No model.pt in {seed_dir}, skipping")
            continue

        output_file = seed_dir / "quality_metrics_highsample.json"
        if output_file.exists():
            print(f"  Already exists: {output_file}, skipping")
            continue

        print(f"  Processing {dataset}/{method}/seed_{seed} ({n_samples} samples)...")

        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed + 9999)  # offset to avoid same seed as training
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + 9999)

        # Create model
        if is_2d:
            model = create_model_2d(
                setup["modes"], setup["hch"], setup["pch"], device
            )
        else:
            model = create_model_1d(
                setup["modes"], setup["width"], setup["mlp_width"], device
            )

        # Create FFM for sampling
        ffm_kwargs = build_ffm_kwargs(config, setup, device)
        ffm = FFMModelOT(model, **ffm_kwargs)

        # Load trained weights
        # model.pt is saved as FNO.state_dict() with keys like "model.fno_blocks..."
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        # Remove non-parameter keys that may be present
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('_')}
        model.load_state_dict(state_dict)
        model.eval()

        # Generate samples
        with torch.no_grad():
            if is_2d:
                spatial_dims = setup["spatial_dims"]
                samples = ffm.sample(list(spatial_dims), n_samples=n_samples).cpu().squeeze()
            else:
                samples = ffm.sample([setup["n_x"]], n_samples=n_samples).cpu().squeeze()

        # Compute MMD-RBF and Sliced Wasserstein
        try:
            mmd = compute_mmd_rbf(ground_truth, samples)
        except Exception as e:
            print(f"    MMD failed: {e}")
            mmd = None

        try:
            sw = compute_sliced_wasserstein(ground_truth, samples)
        except Exception as e:
            print(f"    SW failed: {e}")
            sw = None

        result = {
            "n_samples": n_samples,
            "mmd_rbf": mmd,
            "sliced_wasserstein": sw,
            "seed": seed,
            "method": method,
            "dataset": dataset,
        }

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"    MMD={mmd:.6e}, SW={sw:.6e}" if mmd is not None else "    Failed")

    print("Done.")


if __name__ == "__main__":
    main()
