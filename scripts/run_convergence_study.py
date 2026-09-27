"""
Convergence study: train models and evaluate quality metrics at periodic checkpoints.

This script trains models epoch-by-epoch and periodically generates samples to
compute actual target metrics (mean_mse, variance_mse, autocorrelation_mse /
spectrum_mse_log, plus distributional metrics).  The result is an
`eval_trajectory.json` that records metric values at each checkpoint epoch,
enabling convergence-on-actual-metrics plots.

Usage:
    # Single dataset/kernel
    python run_convergence_study.py --dataset aemet --kernel none --seed 0

    # Baselines
    python run_convergence_study.py --dataset aemet --kernel ddpm --seed 0
    python run_convergence_study.py --dataset navier_stokes --kernel gano --seed 0
"""

import sys
sys.path.append('../')

import argparse
import json
import time
import numpy as np
import torch
import torch.optim as optim
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from torch.utils.data import DataLoader

# Reuse setup functions and config loading from run_seeded_experiments
from run_seeded_experiments import (
    SETUP_FUNCTIONS,
    SPATIAL_2D_DATASETS,
    PDE_DATASETS,
    SEQUENCE_DATASETS,
    DIFFUSION_CONFIGS,
    GANO_CONFIG,
    CFM_OT_CONFIG,
    DIFFUSION_METHODS,
    GAN_METHODS,
    CFM_OT_METHODS,
    BASELINE_METHODS,
    GEOMETRY_METHODS,
    KFFM_RBF_SIGMA_VARIANTS,
    KFFM_SOB_RBF_SIGMA_VARIANTS,
    KFFM_SOB_RBF_PARAM_VARIANTS,
    KFFM_EUC_RBF_CONFIG,
    KFFM_SOB_RBF_CONFIG,
    PDE_TUNED_GEOMETRY_CONFIGS,
    FALLBACK_CONFIGS,
    get_best_config,
    build_ffm_kwargs,
    create_model_1d,
    create_model_2d,
    get_seeds,
)


def get_explicit_geometry_config(dataset: str, kernel: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Return explicit geometry configs that are not discoverable by category lookup."""
    if kernel == 'euc_rbf' or kernel in KFFM_RBF_SIGMA_VARIANTS:
        sigma = KFFM_RBF_SIGMA_VARIANTS.get(kernel, 1.0)
        config = KFFM_EUC_RBF_CONFIG.copy()
        config["ot_kernel_params"] = {"sigma": sigma}
        return f"kFFM-EucRBF(s={sigma})", config

    if kernel == 'sobolev_rbf' or kernel in KFFM_SOB_RBF_SIGMA_VARIANTS or kernel in KFFM_SOB_RBF_PARAM_VARIANTS:
        if dataset not in PDE_DATASETS:
            raise ValueError(f"Sobolev RBF geometry is intended for PDE datasets, got {dataset}")
        if kernel in KFFM_SOB_RBF_PARAM_VARIANTS:
            sigma, sobolev_s = KFFM_SOB_RBF_PARAM_VARIANTS[kernel]
        else:
            sigma = KFFM_SOB_RBF_SIGMA_VARIANTS.get(kernel, 1.0)
            sobolev_s = 1.0
        config = KFFM_SOB_RBF_CONFIG.copy()
        config["ot_kernel_params"] = {"sigma": sigma, "s": sobolev_s, "domain_length": 1.0}
        return f"kFFM-SobolevRBF(sigma={sigma},H{sobolev_s})", config

    if kernel in PDE_TUNED_GEOMETRY_CONFIGS:
        return f"kFFM-tuned({kernel})", deepcopy(PDE_TUNED_GEOMETRY_CONFIGS[kernel])

    return None, None


# ─────────────────────────────────────────────────────────────────────
# Default eval schedule: denser early, sparser later
# ─────────────────────────────────────────────────────────────────────
def default_eval_epochs(total_epochs: int) -> List[int]:
    """Return a list of epochs at which to evaluate.

    Schedule: every 5 epochs for [1..50], every 10 for [60..100],
    every 25 for [125..total_epochs].  Always includes the final epoch.
    """
    eps = list(range(5, min(51, total_epochs + 1), 5))
    eps += list(range(60, min(101, total_epochs + 1), 10))
    eps += list(range(125, total_epochs + 1, 25))
    if total_epochs not in eps:
        eps.append(total_epochs)
    return sorted(set(eps))


# ─────────────────────────────────────────────────────────────────────
# Metric computation at a checkpoint
# ─────────────────────────────────────────────────────────────────────
def evaluate_checkpoint(
    model_or_wrapper,
    ground_truth: torch.Tensor,
    is_2d: bool,
    sample_dims: list,
    n_samples: int = 200,
    compute_distributional: bool = True,
) -> Dict[str, float]:
    """Generate samples and compute quality metrics."""
    from util.eval import (
        GenerationQualityMetrics,
        GenerationQualityMetrics2D,
    )

    samples = model_or_wrapper.sample(
        sample_dims, n_channels=1, n_samples=n_samples,
    ).cpu().squeeze()

    if is_2d:
        qm = GenerationQualityMetrics2D(
            config_name="convergence_study",
            ot_kernel="", ot_method="", ot_coupling="", use_ot=False,
        )
    else:
        qm = GenerationQualityMetrics(
            config_name="convergence_study",
            ot_kernel="", ot_method="", ot_coupling="", use_ot=False,
        )
    qm.compute_from_samples(ground_truth, samples,
                            compute_distributional=compute_distributional)
    d = qm.to_dict()

    # Pick out the metrics we care about
    keys = [
        "mean_mse", "variance_mse",
        "autocorrelation_mse", "spectrum_mse_log",
        "skewness_mse", "kurtosis_mse",
        "mmd_rbf", "sliced_wasserstein", "marginal_wasserstein",
    ]
    return {k: d.get(k) for k in keys if d.get(k) is not None}


# ─────────────────────────────────────────────────────────────────────
# FFM / kFFM convergence study
# ─────────────────────────────────────────────────────────────────────
def run_ffm_convergence(
    dataset: str,
    kernel: str,
    config: dict,
    setup: dict,
    seed: int,
    save_dir: Path,
    device: str,
    train_data: torch.Tensor,
    ground_truth: torch.Tensor,
    eval_epochs: List[int],
    n_eval_samples: int = 200,
) -> Dict[str, Any]:
    """Manual FFM/kFFM training loop with periodic evaluation."""
    from functional_fm_ot import FFMModelOT

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

    batch_size = setup["batch_size_sig"] if kernel == "signature" else setup["batch_size"]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)

    total_epochs = setup["epochs"]
    eval_set = set(eval_epochs)

    trajectory = []
    train_losses = []
    first = True

    for ep in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device).to(torch.float32)
            bs = batch.shape[0]

            if first:
                ffm.n_channels = batch.shape[1]
                ffm.train_dims = batch.shape[2:]
                first = False

            z_noise = ffm.sample_base(bs, ffm.n_channels, ffm.train_dims).to(torch.float32)
            x_data, z_paired = ffm.pair_samples(batch, z_noise)
            t = torch.rand(bs, device=device)
            x_noisy = ffm.simulate(t, x_data, z_paired)
            target = ffm.get_conditional_fields(t, x_data, x_noisy, z_paired)

            x_noisy = x_noisy.to(device)
            target = target.to(device)
            model_out = model(t, x_noisy)

            optimizer.zero_grad()
            loss = torch.mean((model_out - target) ** 2)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

        tr_loss /= len(train_loader)
        train_losses.append(tr_loss)
        scheduler.step()
        epoch_time = time.time() - t0

        ot_tag = "OT" if ffm.use_ot else "indep"
        print(f"  [{ot_tag}] epoch {ep}/{total_epochs} | loss {tr_loss:.6f} | {epoch_time:.1f}s")

        # Evaluate?
        if ep in eval_set:
            print(f"    → evaluating at epoch {ep} ...")
            with torch.no_grad():
                model.eval()
                metrics = evaluate_checkpoint(
                    ffm, ground_truth, is_2d, sample_dims,
                    n_samples=n_eval_samples,
                )
            metrics["epoch"] = ep
            metrics["train_loss"] = tr_loss
            metrics["wall_time"] = epoch_time
            trajectory.append(metrics)
            print(f"      mean_mse={metrics.get('mean_mse', 'N/A'):.4e}  "
                  f"var_mse={metrics.get('variance_mse', 'N/A'):.4e}")

    return {
        "trajectory": trajectory,
        "train_losses": train_losses,
    }


# ─────────────────────────────────────────────────────────────────────
# Diffusion (DDPM / NCSN) convergence study
# ─────────────────────────────────────────────────────────────────────
def run_diffusion_convergence(
    dataset: str,
    method: str,
    setup: dict,
    seed: int,
    save_dir: Path,
    device: str,
    train_data: torch.Tensor,
    ground_truth: torch.Tensor,
    eval_epochs: List[int],
    n_eval_samples: int = 200,
) -> Dict[str, Any]:
    """Manual Diffusion training loop with periodic evaluation."""
    from diffusion import DiffusionModel
    from util.util import make_grid

    is_2d = setup.get("is_2d", False)
    diff_config = DIFFUSION_CONFIGS[method]

    if is_2d:
        model = create_model_2d(setup["modes"], setup["hch"], setup["pch"], device)
        sample_dims = list(setup["spatial_dims"])
    else:
        model = create_model_1d(setup["modes"], setup["width"], setup["mlp_width"], device)
        sample_dims = [setup["n_x"]]

    if diff_config['method'] == 'DDPM':
        diffusion = DiffusionModel(
            model, method='DDPM', T=diff_config['T'], device=device,
            kernel_length=setup["kernel_length"], kernel_variance=setup["kernel_variance"],
            beta_min=diff_config['beta_min'], beta_max=diff_config['beta_max'],
            dtype=torch.float32,
        )
    else:
        diffusion = DiffusionModel(
            model, method='NCSN', T=diff_config['T'], device=device,
            kernel_length=setup["kernel_length"], kernel_variance=setup["kernel_variance"],
            sigma1=diff_config['sigma1'], sigmaT=diff_config['sigmaT'],
            precondition=diff_config.get('precondition', True),
            dtype=torch.float32,
        )

    train_loader = DataLoader(train_data, batch_size=setup["batch_size"], shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)

    T = diffusion.T
    total_epochs = setup["epochs"]
    eval_set = set(eval_epochs)

    trajectory = []
    train_losses = []
    first = True

    for ep in range(1, total_epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0

        for u_0 in train_loader:
            bs = u_0.shape[0]
            u_0 = u_0.to(device).to(torch.float32)

            if first:
                diffusion.n_channels = u_0.shape[1]
                diffusion.train_dims = u_0.shape[2:]
                diffusion.train_support = make_grid(diffusion.train_dims).to(device)
                diffusion.make_loss()
                first = False

            t = torch.randint(1, T + 1, size=[bs], device=device)
            u_t, xi = diffusion.simulate_fwd_process(u_0, t, return_noise=True)
            out = model(t, u_t)

            optimizer.zero_grad()
            loss = diffusion.loss_fxn(xi, out)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

        tr_loss /= len(train_loader)
        train_losses.append(tr_loss)
        scheduler.step()
        epoch_time = time.time() - t0

        print(f"  [{method.upper()}] epoch {ep}/{total_epochs} | loss {tr_loss:.6f} | {epoch_time:.1f}s")

        if ep in eval_set:
            print(f"    → evaluating at epoch {ep} ...")
            with torch.no_grad():
                model.eval()
                diffusion.train_support = make_grid(list(diffusion.train_dims)).to(device)
                metrics = evaluate_checkpoint(
                    diffusion, ground_truth, is_2d, sample_dims,
                    n_samples=n_eval_samples,
                )
            metrics["epoch"] = ep
            metrics["train_loss"] = tr_loss
            metrics["wall_time"] = epoch_time
            trajectory.append(metrics)
            print(f"      mean_mse={metrics.get('mean_mse', 'N/A'):.4e}  "
                  f"var_mse={metrics.get('variance_mse', 'N/A'):.4e}")

    return {
        "trajectory": trajectory,
        "train_losses": train_losses,
    }


# ─────────────────────────────────────────────────────────────────────
# GANO convergence study
# ─────────────────────────────────────────────────────────────────────
def run_gano_convergence(
    dataset: str,
    setup: dict,
    seed: int,
    save_dir: Path,
    device: str,
    train_data: torch.Tensor,
    ground_truth: torch.Tensor,
    eval_epochs: List[int],
    n_eval_samples: int = 200,
) -> Dict[str, Any]:
    """Manual GANO training loop with periodic evaluation."""
    from gano import GANO
    from util.util import make_grid, reshape_channel_last, reshape_channel_first

    is_2d = setup.get("is_2d", False)
    gano_config = GANO_CONFIG.copy()
    factor = gano_config.get('factor', 0.5)

    if is_2d:
        from models.gano_models import Generator, Discriminator
        d_co_domain = gano_config['d_co_domain']
        pad = gano_config['pad']
        in_ch = 1 + 2
        model_g = Generator(in_ch, 1, d_co_domain, pad=pad, factor=factor).to(device)
        model_d = Discriminator(in_ch, 1, d_co_domain, pad=pad, factor=factor).to(device)
        sample_dims = list(setup["spatial_dims"])
    else:
        from models.gano_models import Generator1D, Discriminator1D
        d_co_domain = gano_config.get('d_co_domain_1d', 32)
        pad = gano_config.get('pad_1d', 0)
        in_ch = 1 + 1
        model_g = Generator1D(in_ch, 1, d_co_domain, pad=pad).to(device)
        model_d = Discriminator1D(in_ch, 1, d_co_domain, pad=pad).to(device)
        sample_dims = [setup["n_x"]]

    gano = GANO(
        model_d, model_g,
        l_grad=gano_config['l_grad'], n_critic=gano_config['n_critic'],
        kernel_length=setup["kernel_length"], kernel_variance=setup["kernel_variance"],
        device=device, dtype=torch.float32,
    )

    bs = min(setup["batch_size"], 64) if is_2d else setup["batch_size"]
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)

    opt_g = optim.Adam(model_g.parameters(), lr=1e-3)
    opt_d = optim.Adam(model_d.parameters(), lr=1e-3)
    sched_g = optim.lr_scheduler.StepLR(opt_g, step_size=25, gamma=0.1)
    sched_d = optim.lr_scheduler.StepLR(opt_d, step_size=25, gamma=0.1)

    total_epochs = setup["epochs"]
    eval_set = set(eval_epochs)

    trajectory = []
    g_losses = []
    d_losses = []
    first = True

    for ep in range(1, total_epochs + 1):
        t0 = time.time()
        model_d.train()
        model_g.train()
        d_loss_ep = 0.0
        g_loss_ep = 0.0

        for j, batch in enumerate(train_loader):
            batch = batch.to(device)
            batch_size_cur = batch.shape[0]

            if first:
                gano.n_channels = batch.shape[1]
                gano.train_dims = batch.shape[2:]
                gano.train_support = make_grid(gano.train_dims).to(device)
                first = False

            batch = reshape_channel_last(batch)
            z = gano.gp.sample(gano.train_support, gano.train_dims,
                               n_samples=batch_size_cur, n_channels=gano.n_channels)
            z = reshape_channel_last(z)
            x_syn = model_g(z)

            W_loss = torch.mean(model_d(x_syn.detach())) - torch.mean(model_d(batch))
            gp = gano.calculate_gradient_penalty(batch, x_syn)

            opt_d.zero_grad()
            loss_d = W_loss + gano.l_grad * gp
            loss_d.backward()
            d_loss_ep += loss_d.item()
            opt_d.step()

            if (j + 1) % gano.n_critic == 0:
                opt_g.zero_grad()
                z2 = gano.gp.sample(gano.train_support, gano.train_dims,
                                    n_samples=batch_size_cur, n_channels=gano.n_channels)
                z2 = reshape_channel_last(z2)
                x_syn2 = model_g(z2)
                loss_g = -torch.mean(model_d(x_syn2))
                loss_g.backward()
                g_loss_ep += loss_g.item()
                opt_g.step()

        sched_d.step()
        sched_g.step()
        d_loss_ep /= max(len(train_loader), 1)
        g_loss_ep /= max(len(train_loader), 1)
        d_losses.append(d_loss_ep)
        g_losses.append(g_loss_ep)
        epoch_time = time.time() - t0

        print(f"  [GANO] epoch {ep}/{total_epochs} | D {d_loss_ep:.6f} | G {g_loss_ep:.6f} | {epoch_time:.1f}s")

        if ep in eval_set:
            print(f"    → evaluating at epoch {ep} ...")
            with torch.no_grad():
                model_d.eval()
                model_g.eval()
                metrics = evaluate_checkpoint(
                    gano, ground_truth, is_2d, sample_dims,
                    n_samples=n_eval_samples,
                )
            metrics["epoch"] = ep
            metrics["d_loss"] = d_loss_ep
            metrics["g_loss"] = g_loss_ep
            metrics["wall_time"] = epoch_time
            trajectory.append(metrics)
            print(f"      mean_mse={metrics.get('mean_mse', 'N/A'):.4e}  "
                  f"var_mse={metrics.get('variance_mse', 'N/A'):.4e}")

    return {
        "trajectory": trajectory,
        "d_losses": d_losses,
        "g_losses": g_losses,
    }


# ─────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────
def run_convergence(
    dataset: str,
    kernel: str,
    seed_idx: int,
    output_base: Path,
    base_outputs_dir: Path,
    device: str,
    primary_metric: str = 'mean_mse',
    n_eval_samples: int = 200,
    skip_existing: bool = False,
) -> None:
    """Run one convergence study: one dataset × one kernel × one seed."""

    seeds = get_seeds(10)
    seed = seeds[seed_idx]

    save_dir = output_base / "convergence_study" / dataset / kernel / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    traj_path = save_dir / "eval_trajectory.json"
    if skip_existing and traj_path.exists():
        print(f"  Skipping {dataset}/{kernel}/seed_{seed} (eval_trajectory.json exists)")
        return

    # Setup dataset
    setup_fn = SETUP_FUNCTIONS[dataset]
    setup = setup_fn()

    is_multi = setup.get("is_multi", False)
    is_2d = setup.get("is_2d", False)

    # For multi-dataset (economy), use only the first sub-dataset
    if is_multi:
        first_sub = list(setup["datasets"].keys())[0]
        train_data = setup["datasets"][first_sub]["train_data"]
        ground_truth = setup["datasets"][first_sub]["ground_truth"]
    else:
        train_data = setup["train_data"]
        ground_truth = setup["ground_truth"]

    total_epochs = setup["epochs"]
    eval_epochs = default_eval_epochs(total_epochs)
    print(f"\n{'=' * 70}")
    print(f"Convergence study: {dataset} / {kernel} / seed={seed}")
    print(f"Total epochs: {total_epochs}  |  Eval at: {eval_epochs}")
    print(f"{'=' * 70}")

    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    is_diffusion = kernel in DIFFUSION_METHODS
    is_gano = kernel in GAN_METHODS
    is_cfm_ot = kernel in CFM_OT_METHODS

    t_start = time.time()

    if is_diffusion:
        result = run_diffusion_convergence(
            dataset, kernel, setup, seed, save_dir, device,
            train_data, ground_truth, eval_epochs, n_eval_samples,
        )
    elif is_gano:
        result = run_gano_convergence(
            dataset, setup, seed, save_dir, device,
            train_data, ground_truth, eval_epochs, n_eval_samples,
        )
    else:
        # FFM / kFFM / CFM-OT — need OT config
        if is_cfm_ot:
            config = CFM_OT_CONFIG.copy()
            config_name = "CFM-OT(L2)"
        elif kernel in GEOMETRY_METHODS:
            config_name, config = get_explicit_geometry_config(dataset, kernel)
            if config is None:
                config = FALLBACK_CONFIGS.get(kernel, FALLBACK_CONFIGS['none'])
                config_name = f"{kernel}_fallback"
        else:
            config_name, config = get_best_config(
                dataset, kernel, base_outputs_dir, primary_metric=primary_metric,
            )
            if config is None:
                config = FALLBACK_CONFIGS.get(kernel, FALLBACK_CONFIGS['none'])
                config_name = f"{kernel}_fallback"
        print(f"  OT config: {config_name}")

        result = run_ffm_convergence(
            dataset, kernel, config, setup, seed, save_dir, device,
            train_data, ground_truth, eval_epochs, n_eval_samples,
        )

    total_time = time.time() - t_start

    # Save trajectory
    output = {
        "dataset": dataset,
        "kernel": kernel,
        "seed": seed,
        "seed_idx": seed_idx,
        "total_epochs": total_epochs,
        "eval_epochs": eval_epochs,
        "n_eval_samples": n_eval_samples,
        "total_time_seconds": total_time,
        "trajectory": result["trajectory"],
    }
    if "train_losses" in result:
        output["train_losses"] = result["train_losses"]
    if "d_losses" in result:
        output["d_losses"] = result["d_losses"]
        output["g_losses"] = result["g_losses"]

    with open(traj_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved eval_trajectory.json to {save_dir}")
    print(f"Total wall time: {total_time / 60:.1f} min")


# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Convergence study with periodic metric evaluation")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(SETUP_FUNCTIONS.keys()),
                        help="Dataset name")
    parser.add_argument('--kernel', type=str, required=True,
                        choices=['none', 'signature', 'rbf', 'euclidean'] + BASELINE_METHODS + GEOMETRY_METHODS,
                        help="Kernel or baseline method")
    parser.add_argument('--seed', type=int, required=True,
                        help="Seed index (0-9)")
    parser.add_argument('--metric', type=str, default='mean_mse',
                        help="Primary metric for selecting best config")
    parser.add_argument('--n-eval-samples', type=int, default=200,
                        help="Number of samples to generate at each eval checkpoint")
    parser.add_argument('--skip-existing', action='store_true',
                        help="Skip if eval_trajectory.json already exists")

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    output_base = Path('../outputs/seeded_runs')
    base_outputs_dir = Path('../outputs')

    run_convergence(
        dataset=args.dataset,
        kernel=args.kernel,
        seed_idx=args.seed,
        output_base=output_base,
        base_outputs_dir=base_outputs_dir,
        device=device,
        primary_metric=args.metric,
        n_eval_samples=args.n_eval_samples,
        skip_existing=args.skip_existing,
    )


if __name__ == '__main__':
    main()
