"""
Run experiments with multiple seeds for best OT configurations.

This script allows running experiments on specific datasets with specific
kernels using 10 different seeds to get robust metrics with mean ± std.

Best configurations are **automatically loaded** from existing experiment results:
- comprehensive_metrics.json
- aggregated_results_*.json

This ensures we use the actual best hyperparameters found in sweeps, not hardcoded defaults.

Usage:
    # List available datasets and kernels (shows best configs found)
    python run_seeded_experiments.py --list
    
    # Run specific dataset with specific kernel (10 seeds by default)
    python run_seeded_experiments.py --dataset kdv --kernel euclidean --n_seeds 10
    python run_seeded_experiments.py --dataset aemet --kernel rbf --n_seeds 10
    
    # Run with specific seed range
    python run_seeded_experiments.py --dataset heston --kernel euclidean --seed_start 0 --n_seeds 10
    
    # Run single seed (for parallelization)
    python run_seeded_experiments.py --dataset economy --kernel signature --seed 5
    
    # Show which config will be used for a dataset/kernel
    python run_seeded_experiments.py --dataset aemet --kernel rbf --show-config
    
    # Optimize for a specific metric (default: mean_mse)
    python run_seeded_experiments.py --dataset aemet --kernel signature --metric autocorrelation_mse
    python run_seeded_experiments.py --dataset aemet --kernel signature --metric seasonal_mse
    
    # Available metrics:
    #   mean_mse, variance_mse, autocorrelation_mse, spectrum_mse,
    #   spectrum_mse_log, skewness_mse, kurtosis_mse, seasonal_mse, mmd_rbf
"""

import sys
sys.path.append('../')

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from typing import Dict, Any, List, Tuple, Optional
import importlib.util

# =============================================================================
# Dataset Configuration
# =============================================================================

# Dataset categories for display
PDE_DATASETS = ["kdv", "navier_stokes", "navier_stokes_128", "navier_stokes_v1e5", "stochastic_kdv", "stochastic_ns"]
SEQUENCE_DATASETS = ["aemet", "expr_genes", "economy", "heston", "heston-long"]

# Datasets with 2D spatial structure (for metric computation)
# Note: kdv and stochastic_kdv are 1D spatial, navier_stokes and stochastic_ns are 2D spatial
SPATIAL_2D_DATASETS = ["navier_stokes", "navier_stokes_128", "navier_stokes_v1e5", "stochastic_ns"]

# Dataset to output directory mapping
DATASET_OUTPUT_DIRS = {
    "kdv": "kdv_ot",
    "navier_stokes": "navier_stokes_ot",
    "navier_stokes_128": "navier_stokes_128_ot",
    "navier_stokes_v1e5": "navier_stokes_v1e5_ot",
    "stochastic_kdv": "stochastic_kdv_ot",
    "stochastic_ns": "stochastic_ns_ot",
    "aemet": "AEMET_ot_comprehensive",
    "expr_genes": "expr_genes_ot_comprehensive",
    "economy": "econ_ot_comprehensive",
    "heston": "Heston_ot_kappa1.0",
    "heston-long": "Heston_long_ot_kappa1.0",
}

# For economy, we have subdirectories
ECONOMY_SUBDIRS = {
    "econ1_population": "econ1_population",
    "econ2_gdp": "econ2_gdp",
    "econ3_labor": "econ3_labor",
}

# Kernel categories
KERNEL_CATEGORIES = {
    'none': ['independent'],
    'signature': ['signature_sinkhorn_reg0.1', 'signature_sinkhorn_reg0.5', 'signature_sinkhorn_reg1.0'],
    'rbf': ['rbf_exact', 'rbf_sinkhorn_reg0.1', 'rbf_sinkhorn_reg0.5', 'rbf_sinkhorn_reg1.0'],
    'euclidean': ['euclidean_exact', 'euclidean_sinkhorn_reg0.1', 'euclidean_sinkhorn_reg0.5', 'euclidean_sinkhorn_reg1.0'],
}

# Map ot_kernel values to our kernel categories
OT_KERNEL_TO_CATEGORY = {
    'signature': 'signature',
    'rbf': 'rbf',
    'euclidean': 'euclidean',
    '': 'none',
}

# Diffusion model configurations
DIFFUSION_CONFIGS = {
    'ddpm': {
        'method': 'DDPM',
        'T': 1000,
        'beta_min': 1e-4,
        'beta_max': 0.02,
    },
    'ncsn': {
        'method': 'NCSN',
        'T': 10,
        'sigma1': 1.0,
        'sigmaT': 0.01,
        'precondition': True,
    },
}

# GANO model configuration
GANO_CONFIG = {
    'l_grad': 10,       # Gradient penalty coefficient
    'n_critic': 5,      # Update generator every n_critic iterations
    'pad': 8,           # Padding for non-periodic data (2D)
    'pad_1d': 0,        # Padding for 1D data
    'd_co_domain': 16,  # Hidden channels in GANO (reduced from 32 for memory)
    'd_co_domain_1d': 32,  # Hidden channels for 1D GANO (can be larger)
    'factor': 0.5,      # Channel expansion factor (reduced from 0.75 for memory)
}

# Diffusion methods (not OT kernels, but handled similarly for seeded runs)
DIFFUSION_METHODS = ['ddpm', 'ncsn']

# GAN-based methods (2D only)
GAN_METHODS = ['gano']

# CFM-OT(L2): standard CFM with minibatch OT using L2 cost + i.i.d. Gaussian prior
# (no GP prior, no kernel embedding — the finite-dimensional baseline)
CFM_OT_METHODS = ['cfm_ot']

CFM_OT_CONFIG = {
    "use_ot": True,
    "ot_method": "sinkhorn",
    "ot_reg": 0.1,
    "ot_kernel": "euclidean",
    "ot_coupling": "sample",
    "use_gp_prior": False,  # Standard Gaussian, NOT GP
}

# CFM (independent): i.i.d. Gaussian prior + independent coupling (no OT)
# This is the "no GP, no OT" corner of the 2x2 ablation grid:
#   GP prior + no OT  = FFM (none)       | GP prior + OT  = kFFM (euclidean)
#   Gaussian + no OT  = CFM (cfm_indep)  | Gaussian + OT  = CFM-OT (cfm_ot)
CFM_INDEP_METHODS = ['cfm_indep']

CFM_INDEP_CONFIG = {
    "use_ot": False,
    "use_gp_prior": False,  # Standard Gaussian, NOT GP
}

# CFM-OT with kernel cost (Gaussian prior + kernel OT):
# These decouple the kernel OT benefit from the GP prior.
# If cfm_rbf_ot beats cfm_ot (L2), it proves kernel OT helps independent of GP.
# Sigma-sweep variants for cfm_rbf_ot
CFM_RBF_OT_SIGMA_VARIANTS = {
    'cfm_rbf_ot_s01': 0.1,
    'cfm_rbf_ot_s02': 0.2,
    'cfm_rbf_ot_s05': 0.5,
    'cfm_rbf_ot_s1':  1.0,
    'cfm_rbf_ot_s2':  2.0,
    'cfm_rbf_ot_s10': 10.0,
}

# Reg-sweep variants (sigma from dataset-specific or default, vary ot_reg)
CFM_RBF_OT_REG_VARIANTS = {
    'cfm_rbf_ot_r001': 0.01,
    'cfm_rbf_ot_r005': 0.05,
    'cfm_rbf_ot_r05':  0.5,
    'cfm_rbf_ot_r1':   1.0,
}

# Joint sigma+reg sweep variants: (sigma, reg)
CFM_RBF_OT_JOINT_VARIANTS = {
    'cfm_rbf_ot_s02r001': (0.2, 0.01),
    'cfm_rbf_ot_s05r001': (0.5, 0.01),
    'cfm_rbf_ot_s1r001':  (1.0, 0.01),
    'cfm_rbf_ot_s2r001':  (2.0, 0.01),
    'cfm_rbf_ot_s02r005': (0.2, 0.05),
    'cfm_rbf_ot_s1r005':  (1.0, 0.05),
    'cfm_rbf_ot_s02r05':  (0.2, 0.5),
    'cfm_rbf_ot_s1r05':   (1.0, 0.5),
    'cfm_rbf_ot_s001r01': (0.01, 0.1),    # very small sigma
    'cfm_rbf_ot_s005r01': (0.05, 0.1),    # very small sigma
}

CFM_KERNEL_OT_METHODS = (['cfm_rbf_ot', 'cfm_sig_ot']
                         + list(CFM_RBF_OT_SIGMA_VARIANTS.keys())
                         + list(CFM_RBF_OT_REG_VARIANTS.keys())
                         + list(CFM_RBF_OT_JOINT_VARIANTS.keys()))

CFM_RBF_OT_CONFIG = {
    "use_ot": True,
    "ot_method": "sinkhorn",
    "ot_reg": 0.1,
    "ot_kernel": "rbf",
    "ot_coupling": "sample",
    "use_gp_prior": False,
    "ot_kernel_params": {"sigma": 5.0},
}

# Dataset-specific sigma overrides for cfm_rbf_ot (matched to kFFM-RBF best configs)
CFM_RBF_OT_SIGMA_OVERRIDES = {
    "stochastic_kdv": 0.5,
    "navier_stokes": 1.0,   # try smaller than default 5.0
    "navier_stokes_v1e5": 1.0,  # match standard NS
    "stochastic_ns": 1.0,
}

CFM_SIG_OT_CONFIG = {
    "use_ot": True,
    "ot_method": "sinkhorn",
    "ot_reg": 0.1,
    "ot_kernel": "signature",
    "ot_coupling": "sample",
    "use_gp_prior": False,
    "ot_kernel_params": {
        "time_aug": True,
        "lead_lag": False,
        "dyadic_order": 1,
        "static_kernel_type": "rbf",
        "static_kernel_sigma": 1.0,
        "add_basepoint": True,
        "normalize": True,
        "max_seq_len": 64,
        "max_batch": 32,
    },
}

# Signature geometry follow-up variants for path-space datasets. These keep the
# GP prior and Sinkhorn setup fixed, and only alter the signature-kernel
# hyperparameters used to form the endpoint pairing cost.
SIGNATURE_GEOMETRY_VARIANTS = {
    'signature_ll': {
        "lead_lag": True,
    },
    'signature_d2': {
        "dyadic_order": 2,
        "max_batch": 16,
    },
    'signature_ll_d2': {
        "lead_lag": True,
        "dyadic_order": 2,
        "max_batch": 16,
    },
    'signature_sig05': {
        "static_kernel_sigma": 0.5,
    },
    'signature_sig2': {
        "static_kernel_sigma": 2.0,
    },
    'signature_sig5': {
        "static_kernel_sigma": 5.0,
    },
    'signature_notime': {
        "time_aug": False,
    },
}


def _heston_signature_config(
    ot_reg: float = 0.5,
    static_kernel_sigma: float = 1.0,
    lead_lag: bool = False,
    dyadic_order: int = 1,
    max_seq_len: int = 64,
    time_aug: bool = True,
) -> dict:
    """Explicit Heston signature sweep config for local PySigLib runs."""
    return {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": ot_reg,
        "ot_kernel": "signature",
        "ot_coupling": "sample",
        "use_gp_prior": True,
        "ot_kernel_params": {
            "time_aug": time_aug,
            "lead_lag": lead_lag,
            "dyadic_order": dyadic_order,
            "static_kernel_type": "rbf",
            "static_kernel_sigma": static_kernel_sigma,
            "add_basepoint": True,
            "normalize": True,
            "max_seq_len": max_seq_len,
            "max_batch": 16 if (dyadic_order >= 2 or max_seq_len > 64) else 32,
        },
    }


# Local-only Heston signature sweep. These are deliberately compact because
# true PySigLib runs are currently only available on the local CPU environment.
HESTON_SIGNATURE_SWEEP_CONFIGS = {
    # Current appendix/Table-9-style winner, kept as an anchor.
    'hsig_r05_s1_l64': _heston_signature_config(ot_reg=0.5, static_kernel_sigma=1.0, max_seq_len=64),
    # Use the full 100-point Heston path instead of subsampling to 64 points.
    'hsig_r05_s1_l128': _heston_signature_config(ot_reg=0.5, static_kernel_sigma=1.0, max_seq_len=128),
    # Sinkhorn regularization around the known reg=0.5 winner.
    'hsig_r01_s1_l128': _heston_signature_config(ot_reg=0.1, static_kernel_sigma=1.0, max_seq_len=128),
    'hsig_r1_s1_l128': _heston_signature_config(ot_reg=1.0, static_kernel_sigma=1.0, max_seq_len=128),
    # Static-kernel bandwidth around sigma=1.0 at the known reg=0.5 setting.
    'hsig_r05_s05_l128': _heston_signature_config(ot_reg=0.5, static_kernel_sigma=0.5, max_seq_len=128),
    'hsig_r05_s2_l128': _heston_signature_config(ot_reg=0.5, static_kernel_sigma=2.0, max_seq_len=128),
    # Path-shape variants at the known reg/sigma setting.
    'hsig_r05_s1_ll_l64': _heston_signature_config(ot_reg=0.5, static_kernel_sigma=1.0, lead_lag=True, max_seq_len=64),
    'hsig_r05_s1_d2_l64': _heston_signature_config(ot_reg=0.5, static_kernel_sigma=1.0, dyadic_order=2, max_seq_len=64),
    'hsig_r05_s1_ll_l128': _heston_signature_config(ot_reg=0.5, static_kernel_sigma=1.0, lead_lag=True, max_seq_len=128),
    'hsig_r05_s1_d2_l128': _heston_signature_config(ot_reg=0.5, static_kernel_sigma=1.0, dyadic_order=2, max_seq_len=128),
}

# Explicit geometry-isolation kFFM variants.
# These keep the GP prior, FNO backbone, Sinkhorn solver, and bounded RBF form
# fixed while changing only the metric inside the RBF distance.
KFFM_RBF_SIGMA_VARIANTS = {
    'euc_rbf_s01': 0.1,
    'euc_rbf_s02': 0.2,
    'euc_rbf_s05': 0.5,
    'euc_rbf_s1':  1.0,
    'euc_rbf_s2':  2.0,
    'euc_rbf_s5':  5.0,
    'euc_rbf_s10': 10.0,
    'euc_rbf_s20': 20.0,
}

KFFM_SOB_RBF_SIGMA_VARIANTS = {
    'sobolev_rbf_s05': 0.5,
    'sobolev_rbf_s1':  1.0,
    'sobolev_rbf_s2':  2.0,
    'sobolev_rbf_s5':  5.0,
    'sobolev_rbf_s10': 10.0,
    'sobolev_rbf_s20': 20.0,
}

KFFM_SOB_RBF_PARAM_VARIANTS = {
    'sobolev_h05_s05': (0.5, 0.5),
    'sobolev_h05_s1':  (1.0, 0.5),
    'sobolev_h05_s2':  (2.0, 0.5),
    'sobolev_h05_s5':  (5.0, 0.5),
    'sobolev_h05_s10': (10.0, 0.5),
    'sobolev_h025_s1': (1.0, 0.25),
    'sobolev_h025_s2': (2.0, 0.25),
    'sobolev_h025_s5': (5.0, 0.25),
    'sobolev_h2_s5':   (5.0, 2.0),
    'sobolev_h2_s10':  (10.0, 2.0),
}

KFFM_EUC_RBF_CONFIG = {
    "use_ot": True,
    "ot_method": "sinkhorn",
    "ot_reg": 0.1,
    "ot_kernel": "rbf",
    "ot_coupling": "sample",
    "use_gp_prior": True,
    "ot_kernel_params": {"sigma": 1.0},
}

KFFM_SOB_RBF_CONFIG = {
    "use_ot": True,
    "ot_method": "sinkhorn",
    "ot_reg": 0.1,
    "ot_kernel": "sobolev_rbf",
    "ot_coupling": "sample",
    "use_gp_prior": True,
    "ot_kernel_params": {"sigma": 1.0, "s": 1.0, "domain_length": 1.0},
}


def _pde_rbf_config(
    sigma: float,
    reg: float,
    *,
    kernel_length: Optional[float] = None,
    modes: Optional[int] = None,
    use_gp_prior: bool = True,
) -> dict:
    config = {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": reg,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "use_gp_prior": use_gp_prior,
        "ot_kernel_params": {"sigma": sigma},
    }

    setup_overrides = {}
    if kernel_length is not None:
        setup_overrides["kernel_length"] = kernel_length
    if modes is not None:
        setup_overrides["modes"] = modes
    if setup_overrides:
        config["setup_overrides"] = setup_overrides

    return config


PDE_TUNED_GEOMETRY_CONFIGS = {
    # Stochastic NS candidates from the earlier spectrum sweep, promoted to
    # seeded MMD runs. These keep the GP prior but tune prior length and use a
    # non-linear endpoint cost.
    'sns_gp005_rbf_s20_r005': {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.05,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "use_gp_prior": True,
        "ot_kernel_params": {"sigma": 20.0},
        "setup_overrides": {"kernel_length": 0.05},
    },
    'sns_gp01_rbf_s20_r005': {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.05,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "use_gp_prior": True,
        "ot_kernel_params": {"sigma": 20.0},
        "setup_overrides": {"kernel_length": 0.1},
    },
    'sns_gp01_rbf_s30_r005': {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.05,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "use_gp_prior": True,
        "ot_kernel_params": {"sigma": 30.0},
        "setup_overrides": {"kernel_length": 0.1},
    },
    'sns_gp005_sob_h05_s20_r005': {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.05,
        "ot_kernel": "sobolev_rbf",
        "ot_coupling": "sample",
        "use_gp_prior": True,
        "ot_kernel_params": {"sigma": 20.0, "s": 0.5, "domain_length": 1.0},
        "setup_overrides": {"kernel_length": 0.05},
    },
    'sns_gp01_sob_h05_s20_r005': {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.05,
        "ot_kernel": "sobolev_rbf",
        "ot_coupling": "sample",
        "use_gp_prior": True,
        "ot_kernel_params": {"sigma": 20.0, "s": 0.5, "domain_length": 1.0},
        "setup_overrides": {"kernel_length": 0.1},
    },
    # RBF-only expansion for the hard 2D stochastic NS row. These sweep around
    # the previous best RBF setting (GP length 0.05, sigma 20, reg 0.05) and
    # around the iid CFM-RBF Table 1 winner.
    'sns_gp003_rbf_s15_r002': _pde_rbf_config(15.0, 0.02, kernel_length=0.03),
    'sns_gp003_rbf_s20_r002': _pde_rbf_config(20.0, 0.02, kernel_length=0.03),
    'sns_gp005_rbf_s15_r002': _pde_rbf_config(15.0, 0.02, kernel_length=0.05),
    'sns_gp005_rbf_s20_r002': _pde_rbf_config(20.0, 0.02, kernel_length=0.05),
    'sns_gp005_rbf_s20_r01': _pde_rbf_config(20.0, 0.1, kernel_length=0.05),
    'sns_gp005_rbf_s30_r002': _pde_rbf_config(30.0, 0.02, kernel_length=0.05),
    'sns_gp0075_rbf_s20_r002': _pde_rbf_config(20.0, 0.02, kernel_length=0.075),
    'sns_iid_rbf_s025_r005': _pde_rbf_config(0.25, 0.05, use_gp_prior=False),
    'sns_iid_rbf_s05_r005': _pde_rbf_config(0.5, 0.05, use_gp_prior=False),
    'sns_iid_rbf_s05_r002': _pde_rbf_config(0.5, 0.02, use_gp_prior=False),
    'sns_iid_rbf_s1_r005': _pde_rbf_config(1.0, 0.05, use_gp_prior=False),
    # Deterministic Navier-Stokes candidates from the old focus sweep. The RBF
    # variants are the strongest non-Euclidean leads; Sobolev companions test
    # whether function-space weighting can explain the same gains.
    'ns_gp001_rbf_s10_m24_r05': {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.5,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "use_gp_prior": True,
        "ot_kernel_params": {"sigma": 10.0},
        "setup_overrides": {"kernel_length": 0.01, "modes": 24},
    },
    'ns_gp002_rbf_s20_m32_r05': {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.5,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "use_gp_prior": True,
        "ot_kernel_params": {"sigma": 20.0},
        "setup_overrides": {"kernel_length": 0.02, "modes": 32},
    },
    'ns_gp005_rbf_s20_r01': {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.1,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "use_gp_prior": True,
        "ot_kernel_params": {"sigma": 20.0},
        "setup_overrides": {"kernel_length": 0.05},
    },
    'ns_gp001_sob_h05_s10_m24_r05': {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.5,
        "ot_kernel": "sobolev_rbf",
        "ot_coupling": "sample",
        "use_gp_prior": True,
        "ot_kernel_params": {"sigma": 10.0, "s": 0.5, "domain_length": 1.0},
        "setup_overrides": {"kernel_length": 0.01, "modes": 24},
    },
    'ns_gp002_sob_h05_s20_m32_r05': {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.5,
        "ot_kernel": "sobolev_rbf",
        "ot_coupling": "sample",
        "use_gp_prior": True,
        "ot_kernel_params": {"sigma": 20.0, "s": 0.5, "domain_length": 1.0},
        "setup_overrides": {"kernel_length": 0.02, "modes": 32},
    },
    'ns_gp005_sob_h05_s20_r01': {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.1,
        "ot_kernel": "sobolev_rbf",
        "ot_coupling": "sample",
        "use_gp_prior": True,
        "ot_kernel_params": {"sigma": 20.0, "s": 0.5, "domain_length": 1.0},
        "setup_overrides": {"kernel_length": 0.05},
    },
    # RBF-only expansion for the deterministic Navier-Stokes row. These sweep
    # around the best completed tuned RBF setting (GP length 0.05, sigma 20,
    # reg 0.1) and around the iid CFM-RBF Table 1 winner.
    'ns_gp003_rbf_s20_r005': _pde_rbf_config(20.0, 0.05, kernel_length=0.03),
    'ns_gp003_rbf_s20_r01': _pde_rbf_config(20.0, 0.1, kernel_length=0.03),
    'ns_gp005_rbf_s15_r005': _pde_rbf_config(15.0, 0.05, kernel_length=0.05),
    'ns_gp005_rbf_s15_r01': _pde_rbf_config(15.0, 0.1, kernel_length=0.05),
    'ns_gp005_rbf_s20_r005': _pde_rbf_config(20.0, 0.05, kernel_length=0.05),
    'ns_gp005_rbf_s30_r005': _pde_rbf_config(30.0, 0.05, kernel_length=0.05),
    'ns_gp0075_rbf_s20_r005': _pde_rbf_config(20.0, 0.05, kernel_length=0.075),
    'ns_gp005_rbf_s20_m24_r005': _pde_rbf_config(20.0, 0.05, kernel_length=0.05, modes=24),
    'ns_iid_rbf_s05_r001': _pde_rbf_config(0.5, 0.01, use_gp_prior=False),
    'ns_iid_rbf_s1_r005': _pde_rbf_config(1.0, 0.005, use_gp_prior=False),
    'ns_iid_rbf_s1_r002': _pde_rbf_config(1.0, 0.02, use_gp_prior=False),
    'ns_iid_rbf_s2_r001': _pde_rbf_config(2.0, 0.01, use_gp_prior=False),
}

GEOMETRY_METHODS = (['euc_rbf', 'sobolev_rbf']
                    + list(SIGNATURE_GEOMETRY_VARIANTS.keys())
                    + list(HESTON_SIGNATURE_SWEEP_CONFIGS.keys())
                    + list(KFFM_RBF_SIGMA_VARIANTS.keys())
                    + list(KFFM_SOB_RBF_SIGMA_VARIANTS.keys())
                    + list(KFFM_SOB_RBF_PARAM_VARIANTS.keys())
                    + list(PDE_TUNED_GEOMETRY_CONFIGS.keys()))

# All baseline methods (non-OT-kernel)
BASELINE_METHODS = (DIFFUSION_METHODS + GAN_METHODS + CFM_OT_METHODS
                    + CFM_INDEP_METHODS + CFM_KERNEL_OT_METHODS
                    + GEOMETRY_METHODS)

# =============================================================================
# Load Best Configurations from Experiment Results
# =============================================================================

def load_experiment_results(dataset: str, outputs_dir: Path) -> Optional[Dict]:
    """
    Load experiment results for a dataset from comprehensive_metrics.json
    or aggregated_results.json files.
    
    Returns a dict of config_name -> config_data (including metrics and OT params)
    """
    dataset_dir = outputs_dir / DATASET_OUTPUT_DIRS.get(dataset, dataset)
    
    if not dataset_dir.exists():
        return None
    
    # For economy, look in the econ1_population subdirectory
    if dataset == "economy":
        dataset_dir = dataset_dir / "econ1_population"
    
    configs = {}
    
    # Try aggregated_results first (includes sweep experiments)
    agg_patterns = [
        f"aggregated_results_*.json",
        "aggregated_results.json",
    ]
    for pattern in agg_patterns:
        for agg_file in dataset_dir.glob(pattern):
            try:
                with open(agg_file, 'r') as f:
                    data = json.load(f)
                if 'configs' in data:
                    for name, cfg in data['configs'].items():
                        if 'metrics' in cfg:
                            configs[name] = cfg['metrics']
                        else:
                            configs[name] = cfg
            except (json.JSONDecodeError, IOError):
                continue
    
    # Also try parent directory for aggregated results
    parent_dir = dataset_dir.parent
    for pattern in agg_patterns:
        for agg_file in parent_dir.glob(pattern):
            try:
                with open(agg_file, 'r') as f:
                    data = json.load(f)
                if 'configs' in data:
                    for name, cfg in data['configs'].items():
                        if name not in configs:  # Don't overwrite
                            if 'metrics' in cfg:
                                configs[name] = cfg['metrics']
                            else:
                                configs[name] = cfg
            except (json.JSONDecodeError, IOError):
                continue
    
    # Try comprehensive_metrics.json
    for metrics_file in [dataset_dir / "comprehensive_metrics.json", 
                         parent_dir / "comprehensive_metrics.json"]:
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                if 'configs' in data:
                    cfg_list = data['configs']
                    if isinstance(cfg_list, list):
                        for cfg in cfg_list:
                            name = cfg.get('config_name', '')
                            # Remove dataset prefix if present
                            for prefix in ['econ1_population_', 'econ2_population_', 'econ3_population_']:
                                if name.startswith(prefix):
                                    name = name[len(prefix):]
                                    break
                            if name and name not in configs:
                                configs[name] = cfg
                    elif isinstance(cfg_list, dict):
                        for name, cfg in cfg_list.items():
                            if name not in configs:
                                configs[name] = cfg
            except (json.JSONDecodeError, IOError):
                continue
    
    return configs if configs else None


def categorize_config(config_name: str, config_data: Dict) -> Optional[str]:
    """
    Determine which kernel category a config belongs to.
    Returns: 'none', 'signature', 'rbf', 'euclidean', or None if not categorizable.
    """
    # Skip baselines
    if config_name in ['DDPM', 'NCSN']:
        return None
    
    # Independent = no OT
    if config_name == 'independent' or not config_data.get('use_ot', False):
        return 'none'
    
    # Skip gaussian OT
    if config_data.get('ot_method') == 'gaussian':
        return None
    
    # Use ot_kernel field (most reliable)
    ot_kernel = config_data.get('ot_kernel', '').lower()
    if ot_kernel in OT_KERNEL_TO_CATEGORY:
        return OT_KERNEL_TO_CATEGORY[ot_kernel]
    
    # Fallback: infer from config name
    name_lower = config_name.lower()
    if 'signature' in name_lower or name_lower.startswith('sig_'):
        return 'signature'
    elif 'rbf' in name_lower:
        return 'rbf'
    elif 'euclidean' in name_lower:
        return 'euclidean'
    
    return None


def find_best_config_for_kernel(
    configs: Dict,
    kernel_category: str,
    primary_metric: str = 'mean_mse',
    dataset_type: str = 'sequence'
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Find the best config for a kernel category based on the primary metric.
    
    Args:
        configs: Dict of config_name -> config_data
        kernel_category: 'none', 'signature', 'rbf', or 'euclidean'
        primary_metric: Metric to optimize (default: 'mean_mse')
        dataset_type: 'sequence' or 'pde' (affects metric selection)
    
    Returns:
        (best_config_name, best_config_data) or (None, None) if not found
    """
    # Determine metrics to consider for ranking
    if dataset_type == 'pde':
        ranking_metrics = ['mean_mse', 'variance_mse', 'spectrum_mse_log']
    else:
        ranking_metrics = ['mean_mse', 'variance_mse', 'autocorrelation_mse']
    
    candidates = []
    
    for config_name, config_data in configs.items():
        category = categorize_config(config_name, config_data)
        if category != kernel_category:
            continue
        
        # Get metric value
        metric_val = config_data.get(primary_metric)
        if metric_val is None or not isinstance(metric_val, (int, float)):
            continue
        
        candidates.append((config_name, config_data, metric_val))
    
    if not candidates:
        return None, None
    
    # Sort by primary metric (lower is better)
    candidates.sort(key=lambda x: x[2])
    best_name, best_data, _ = candidates[0]
    
    return best_name, best_data


def extract_ot_config(config_data: Dict) -> Dict:
    """
    Extract OT configuration parameters from config data.
    Returns a dict suitable for passing to FFMModelOT.
    """
    if not config_data.get('use_ot', False):
        return {"use_ot": False}
    
    ot_config = {
        "use_ot": True,
        "ot_method": config_data.get('ot_method', 'sinkhorn'),
        "ot_kernel": config_data.get('ot_kernel', 'euclidean'),
        "ot_coupling": config_data.get('ot_coupling', 'sample'),
    }
    
    # Add regularization if present
    if 'ot_reg' in config_data:
        ot_config['ot_reg'] = config_data['ot_reg']
    else:
        # Try to infer from config name
        config_name = config_data.get('config_name', '')
        if 'reg0.1' in config_name:
            ot_config['ot_reg'] = 0.1
        elif 'reg0.5' in config_name:
            ot_config['ot_reg'] = 0.5
        elif 'reg1.0' in config_name:
            ot_config['ot_reg'] = 1.0
        elif 'reg2.0' in config_name:
            ot_config['ot_reg'] = 2.0
        elif 'reg5.0' in config_name:
            ot_config['ot_reg'] = 5.0
        else:
            ot_config['ot_reg'] = 0.1  # default
    
    # Add kernel params based on kernel type
    kernel = ot_config['ot_kernel']
    
    if kernel == 'rbf':
        # Try to extract sigma from config name or use default
        config_name = config_data.get('config_name', '')
        sigma = 5.0  # default
        if 'sigma0.1' in config_name:
            sigma = 0.1
        elif 'sigma0.2' in config_name:
            sigma = 0.2
        elif 'sigma0.5' in config_name:
            sigma = 0.5
        elif 'sigma1.0' in config_name:
            sigma = 1.0
        elif 'sigma2.0' in config_name:
            sigma = 2.0
        elif 'sigma5.0' in config_name:
            sigma = 5.0
        ot_config['ot_kernel_params'] = {"sigma": sigma}
    
    elif kernel == 'signature':
        # Default signature params
        ot_config['ot_kernel_params'] = {
            "time_aug": True,
            "lead_lag": False,
            "dyadic_order": 1,
            "static_kernel_type": "rbf",
            "static_kernel_sigma": 1.0,
            "add_basepoint": True,
            "normalize": True,
            "max_seq_len": 64,
            "max_batch": 32,
        }
        # Try to extract params from config name
        config_name = config_data.get('config_name', '')
        if 'leadlag' in config_name.lower():
            ot_config['ot_kernel_params']['lead_lag'] = True
        if 'order2' in config_name:
            ot_config['ot_kernel_params']['dyadic_order'] = 2
        elif 'order3' in config_name:
            ot_config['ot_kernel_params']['dyadic_order'] = 3
        if 'no_normalize' in config_name:
            ot_config['ot_kernel_params']['normalize'] = False
        if 'no_time_aug' in config_name:
            ot_config['ot_kernel_params']['time_aug'] = False
        # Extract static kernel sigma
        if 'sigma0.5' in config_name:
            ot_config['ot_kernel_params']['static_kernel_sigma'] = 0.5
        elif 'sigma2.0' in config_name:
            ot_config['ot_kernel_params']['static_kernel_sigma'] = 2.0
        elif 'sigma5.0' in config_name:
            ot_config['ot_kernel_params']['static_kernel_sigma'] = 5.0
    
    return ot_config


def get_best_config(
    dataset: str,
    kernel: str,
    outputs_dir: Path,
    primary_metric: str = 'mean_mse',
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Get the best config for a dataset/kernel combination.
    
    Parameters
    ----------
    dataset : str
        Dataset name
    kernel : str
        Kernel category ('none', 'signature', 'rbf', 'euclidean')
    outputs_dir : Path
        Base outputs directory
    primary_metric : str
        Metric to optimize for (default: 'mean_mse')
        Options: mean_mse, variance_mse, autocorrelation_mse, spectrum_mse, 
                 spectrum_mse_log, skewness_mse, kurtosis_mse, seasonal_mse
    
    Returns:
        (config_name, ot_config) or (None, None) if not found
    """
    configs = load_experiment_results(dataset, outputs_dir)

    if not configs and dataset == "navier_stokes_v1e5":
        # No sweep exists for the turbulent regime; borrow the tuned configs
        # from the standard 64x64 NS dataset (same grid, FNO, and GP prior),
        # so that only the data regime changes.
        print("  navier_stokes_v1e5: borrowing best configs from navier_stokes")
        configs = load_experiment_results("navier_stokes", outputs_dir)

    if not configs:
        print(f"  Warning: No experiment results found for {dataset}")
        return None, None
    
    # Determine dataset type
    dataset_type = 'pde' if dataset in PDE_DATASETS else 'sequence'
    
    # Find best config
    best_name, best_data = find_best_config_for_kernel(
        configs, kernel, 
        primary_metric=primary_metric,
        dataset_type=dataset_type
    )
    
    if best_name is None:
        return None, None
    
    # Extract OT config
    ot_config = extract_ot_config(best_data)
    ot_config['_source_config'] = best_name  # Store source for reference
    ot_config['_source_metrics'] = {
        k: v for k, v in best_data.items() 
        if k in ['mean_mse', 'variance_mse', 'autocorrelation_mse', 'spectrum_mse_log',
                 'spectrum_mse', 'skewness_mse', 'kurtosis_mse', 'seasonal_mse',
                 'mmd_rbf']
    }
    ot_config['_optimized_metric'] = primary_metric
    
    return best_name, ot_config


# =============================================================================
# Fallback Configs (used when no experiment results available)
# =============================================================================

FALLBACK_CONFIGS = {
    "none": {"use_ot": False},
    "signature": {
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
            "static_kernel_sigma": 1.0,
            "add_basepoint": True,
            "normalize": True,
            "max_seq_len": 64,
            "max_batch": 32,
        },
    },
    "rbf": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.1,
        "ot_kernel": "rbf",
        "ot_coupling": "sample",
        "ot_kernel_params": {"sigma": 5.0},
    },
    "euclidean": {
        "use_ot": True,
        "ot_method": "sinkhorn",
        "ot_reg": 0.1,
        "ot_kernel": "euclidean",
        "ot_coupling": "sample",
    },
}

# =============================================================================
# Utility Functions
# =============================================================================

def list_available_options(outputs_dir: Path):
    """Print available datasets and kernels with best configs found."""
    print("\n" + "=" * 70)
    print("Available Datasets and Kernels/Methods")
    print("=" * 70)
    
    all_kernels = ['none', 'signature', 'rbf', 'euclidean']
    
    print("\n📊 PDE Datasets:")
    for ds in PDE_DATASETS:
        print(f"\n  {ds.upper()}:")
        for kernel in all_kernels:
            if kernel == 'signature' and ds in ['navier_stokes', 'stochastic_ns']:
                print(f"    • {kernel}: N/A (2D data)")
                continue
            name, config = get_best_config(ds, kernel, outputs_dir)
            if name:
                metrics = config.get('_source_metrics', {})
                mean_mse = metrics.get('mean_mse', 'N/A')
                if isinstance(mean_mse, float):
                    mean_mse = f"{mean_mse:.2e}"
                print(f"    • {kernel}: {name} (mean_mse={mean_mse})")
            else:
                print(f"    • {kernel}: [using fallback config]")
        # Show diffusion baselines
        for method in DIFFUSION_METHODS:
            config = DIFFUSION_CONFIGS[method]
            print(f"    • {method}: {config['method']} (T={config['T']})")
        # Show GANO for all datasets
        print(f"    • gano: GANO (l_grad={GANO_CONFIG['l_grad']}, n_critic={GANO_CONFIG['n_critic']})")
    
    print("\n📈 Sequence Datasets:")
    for ds in SEQUENCE_DATASETS:
        print(f"\n  {ds.upper()}:")
        for kernel in all_kernels:
            name, config = get_best_config(ds, kernel, outputs_dir)
            if name:
                metrics = config.get('_source_metrics', {})
                mean_mse = metrics.get('mean_mse', 'N/A')
                if isinstance(mean_mse, float):
                    mean_mse = f"{mean_mse:.2e}"
                print(f"    • {kernel}: {name} (mean_mse={mean_mse})")
            else:
                print(f"    • {kernel}: [using fallback config]")
        # Show diffusion baselines (DDPM only for 1D)
        print(f"    • ddpm: DDPM (T={DIFFUSION_CONFIGS['ddpm']['T']})")
        # Show GANO for all datasets
        print(f"    • gano: GANO (l_grad={GANO_CONFIG['l_grad']}, n_critic={GANO_CONFIG['n_critic']})")
    
    print("\n" + "=" * 70)
    print("Note: Best configs are loaded from existing experiment results.")
    print("If no results found, fallback configs with default hyperparameters are used.")
    print("Diffusion baselines (ddpm, ncsn) use fixed configs.")
    print("GANO is only available for 2D PDE datasets (navier_stokes, stochastic_ns).")
    print("=" * 70)
    print()


def show_config(dataset: str, kernel: str, outputs_dir: Path, primary_metric: str = 'mean_mse'):
    """Show the config that will be used for a dataset/kernel combination."""
    print("\n" + "=" * 70)
    print(f"Config for {dataset.upper()} / {kernel.upper()}")
    print(f"Optimizing for: {primary_metric}")
    print("=" * 70)

    explicit_config = None
    explicit_name = None
    if kernel in SIGNATURE_GEOMETRY_VARIANTS:
        explicit_name = f"kFFM-Signature({kernel})"
        explicit_config = FALLBACK_CONFIGS["signature"].copy()
        params = FALLBACK_CONFIGS["signature"]["ot_kernel_params"].copy()
        params.update(SIGNATURE_GEOMETRY_VARIANTS[kernel])
        explicit_config["ot_kernel_params"] = params
        explicit_config["use_gp_prior"] = True
    elif kernel in HESTON_SIGNATURE_SWEEP_CONFIGS:
        explicit_name = f"kFFM-HestonSignature({kernel})"
        explicit_config = deepcopy(HESTON_SIGNATURE_SWEEP_CONFIGS[kernel])
    elif kernel == 'euc_rbf' or kernel in KFFM_RBF_SIGMA_VARIANTS:
        sigma = KFFM_RBF_SIGMA_VARIANTS.get(kernel, 1.0)
        explicit_name = f"kFFM-EucRBF(s={sigma})"
        explicit_config = KFFM_EUC_RBF_CONFIG.copy()
        explicit_config["ot_kernel_params"] = {"sigma": sigma}
    elif kernel == 'sobolev_rbf' or kernel in KFFM_SOB_RBF_SIGMA_VARIANTS or kernel in KFFM_SOB_RBF_PARAM_VARIANTS:
        if kernel in KFFM_SOB_RBF_PARAM_VARIANTS:
            sigma, sobolev_s = KFFM_SOB_RBF_PARAM_VARIANTS[kernel]
        else:
            sigma = KFFM_SOB_RBF_SIGMA_VARIANTS.get(kernel, 1.0)
            sobolev_s = 1.0
        explicit_name = f"kFFM-SobolevRBF(sigma={sigma},H{sobolev_s})"
        explicit_config = KFFM_SOB_RBF_CONFIG.copy()
        explicit_config["ot_kernel_params"] = {"sigma": sigma, "s": sobolev_s, "domain_length": 1.0}
    elif kernel in PDE_TUNED_GEOMETRY_CONFIGS:
        explicit_name = f"kFFM-tuned({kernel})"
        explicit_config = deepcopy(PDE_TUNED_GEOMETRY_CONFIGS[kernel])

    if explicit_config is not None:
        print(f"\nExplicit config: {explicit_name}")
        print(f"\nOT Configuration:")
        for k, v in explicit_config.items():
            if not k.startswith('_'):
                print(f"  {k}: {v}")
        print()
        return
    
    name, config = get_best_config(dataset, kernel, outputs_dir, primary_metric=primary_metric)
    
    if name:
        print(f"\n✓ Best config found: {name}")
        print(f"\nSource metrics:")
        for k, v in config.get('_source_metrics', {}).items():
            marker = " ← optimized" if k == primary_metric else ""
            if isinstance(v, float):
                print(f"  {k}: {v:.2e}{marker}")
            else:
                print(f"  {k}: {v}{marker}")
    else:
        print(f"\n⚠ No experiment results found, using fallback config")
        config = FALLBACK_CONFIGS.get(kernel, FALLBACK_CONFIGS['none'])
    
    print(f"\nOT Configuration:")
    for k, v in config.items():
        if not k.startswith('_'):
            print(f"  {k}: {v}")
    
    print()


def get_seeds(n_seeds: int, seed_start: int = 0) -> List[int]:
    """Generate seed list."""
    return [2**i for i in range(seed_start, seed_start + n_seeds)]


def load_quality_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load quality metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def aggregate_metrics(
    metrics_list: List[Dict[str, Any]],
    dataset: str,
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across seeds to compute mean ± std.
    
    Returns dict with metric_name -> {mean, std, values}
    """
    # Determine which metrics to aggregate based on dataset type
    is_pde = dataset in PDE_DATASETS
    
    if is_pde:
        metric_keys = ["mmd_rbf", "mean_mse", "variance_mse", "spectrum_mse_log"]
    else:
        metric_keys = ["mmd_rbf", "mean_mse", "variance_mse", "autocorrelation_mse"]
    
    aggregated = {}
    for key in metric_keys:
        values = []
        for m in metrics_list:
            if key in m and m[key] is not None:
                # Filter out None values and NaN values
                val = m[key]
                if not (isinstance(val, float) and np.isnan(val)):
                    values.append(val)
        
        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
                "n_seeds": len(values),
            }
    
    return aggregated


def format_metric(mean: float, std: float) -> str:
    """Format metric as mean ± std in scientific notation."""
    # Determine the exponent
    if mean == 0:
        return "0.00 ± 0.00"
    
    exp = int(np.floor(np.log10(abs(mean))))
    mantissa_mean = mean / (10 ** exp)
    mantissa_std = std / (10 ** exp)
    
    return f"{mantissa_mean:.2f} ± {mantissa_std:.2f} × 10^{exp}"


def print_results(
    aggregated: Dict[str, Dict[str, float]],
    dataset: str,
    kernel: str,
):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print(f"Results: {dataset.upper()} with {kernel.upper()} kernel")
    print("=" * 70)
    
    is_pde = dataset in PDE_DATASETS
    
    if is_pde:
        headers = ["MMD-RBF", "Mean MSE", "Variance MSE", "Spectrum (log) MSE"]
        keys = ["mmd_rbf", "mean_mse", "variance_mse", "spectrum_mse_log"]
    else:
        headers = ["MMD-RBF", "Mean MSE", "Variance MSE", "Autocorr MSE"]
        keys = ["mmd_rbf", "mean_mse", "variance_mse", "autocorrelation_mse"]
    
    print(f"\n{'Metric':<25} {'Mean ± Std':<35} {'N Seeds':<10}")
    print("-" * 70)
    
    for header, key in zip(headers, keys):
        if key in aggregated:
            m = aggregated[key]
            formatted = format_metric(m["mean"], m["std"])
            print(f"{header:<25} {formatted:<35} {m['n_seeds']:<10}")
        else:
            print(f"{header:<25} {'N/A':<35} {'0':<10}")
    
    print()


# =============================================================================
# Dataset-Specific Training Functions
# =============================================================================

def setup_kdv():
    """Setup KdV dataset and model."""
    from util.util import load_kdv
    from torch.utils.data import DataLoader
    
    data = load_kdv('../data/KdV.mat', mode='snapshot')
    n_total = data.shape[0]
    idx = torch.randperm(n_total)
    data = data[idx]
    ntr = min(160, int(0.8 * n_total))
    train_data = data[:ntr]
    ground_truth = train_data.squeeze(1).clone()
    n_x = train_data.shape[-1]
    
    return {
        "train_data": train_data,
        "ground_truth": ground_truth,
        "n_x": n_x,
        "batch_size": 16,
        "batch_size_sig": 16,
        "modes": 64,
        "width": 256,
        "mlp_width": 128,
        "kernel_length": 0.01,
        "kernel_variance": 0.1,
        "epochs": 300,
        "n_gen_samples": 200,
        "is_2d": False,
    }


def setup_navier_stokes():
    """Setup Navier-Stokes dataset."""
    from util.util import load_navier_stokes
    
    data = load_navier_stokes('../data/ns.mat', shuffle=True, subsample_time=5)
    n_total = data.shape[0]
    ntr = min(20000, int(0.8 * n_total))
    train_data = data[:ntr]
    ground_truth = train_data.squeeze(1).clone()
    spatial_dims = train_data.shape[2:]
    
    return {
        "train_data": train_data,
        "ground_truth": ground_truth[:1000],
        "spatial_dims": spatial_dims,
        "batch_size": 512,
        "batch_size_sig": 512,
        "modes": 16,
        "hch": 32,
        "pch": 64,
        "kernel_length": 0.001,
        "kernel_variance": 1.0,
        "epochs": 300,
        "n_gen_samples": 100,
        "is_2d": True,
    }


def setup_navier_stokes_128():
    """Setup 128x128 Navier-Stokes dataset (neuraloperator/Zenodo).

    Data from https://zenodo.org/records/12825163 (nsforcing_128.tgz).
    Higher spatial resolution than the standard 64x64 dataset.
    """
    from util.util import load_navier_stokes_128

    data = load_navier_stokes_128('../data/nsforcing_128', shuffle=True)
    n_total = data.shape[0]
    ntr = min(20000, int(0.8 * n_total))
    train_data = data[:ntr]
    ground_truth = train_data.squeeze(1).clone()
    spatial_dims = train_data.shape[2:]  # (128, 128)

    return {
        "train_data": train_data,
        "ground_truth": ground_truth[:1000],
        "spatial_dims": spatial_dims,
        "batch_size": 256,       # Smaller batch for 128x128 (4x more pixels)
        "batch_size_sig": 256,
        "modes": 24,             # More Fourier modes for higher resolution
        "hch": 32,
        "pch": 64,
        "kernel_length": 0.001,
        "kernel_variance": 1.0,
        "epochs": 300,
        "n_gen_samples": 100,
        "is_2d": True,
    }


def setup_navier_stokes_v1e5():
    """Setup turbulent Navier-Stokes dataset (viscosity 1e-5, original FNO benchmark).

    Data: data/NavierStokes_V1e-5_N1200_T20.mat. Only spun-up snapshots
    (t >= 10) are used so the dataset sits in the fully turbulent regime.
    Same FNO/GP configuration as the standard 64x64 NS dataset so that the
    only change relative to `navier_stokes` is the data regime.
    """
    from util.util import load_navier_stokes_v1e5

    data = load_navier_stokes_v1e5('../data/NavierStokes_V1e-5_N1200_T20.mat',
                                   shuffle=True, time_start=10)
    n_total = data.shape[0]
    ntr = min(20000, int(0.8 * n_total))
    train_data = data[:ntr]
    ground_truth = train_data.squeeze(1).clone()
    spatial_dims = train_data.shape[2:]

    return {
        "train_data": train_data,
        "ground_truth": ground_truth[:1000],
        "spatial_dims": spatial_dims,
        "batch_size": 512,
        "batch_size_sig": 512,
        "modes": 16,
        "hch": 32,
        "pch": 64,
        "kernel_length": 0.001,
        "kernel_variance": 1.0,
        "epochs": 300,
        "n_gen_samples": 100,
        "is_2d": True,
    }


def setup_stochastic_kdv():
    """Setup Stochastic KdV dataset."""
    from util.util import load_stochastic_kdv
    
    data = load_stochastic_kdv('../data/stochastic_kdv.mat', shuffle=True, mode='snapshot')
    n_total = data.shape[0]
    ntr = min(50000, int(0.8 * n_total))
    train_data = data[:ntr]
    ground_truth = train_data.squeeze(1).clone()
    n_x = train_data.shape[-1]
    
    return {
        "train_data": train_data,
        "ground_truth": ground_truth,
        "n_x": n_x,
        "batch_size": 512,
        "batch_size_sig": 128,
        "modes": 32,
        "width": 256,
        "mlp_width": 128,
        "kernel_length": 0.01,
        "kernel_variance": 0.1,
        "epochs": 100,
        "n_gen_samples": 500,
        "is_2d": False,
    }


def setup_stochastic_ns():
    """Setup Stochastic NS dataset."""
    from util.util import load_stochastic_ns
    
    data = load_stochastic_ns('../data/stochastic_ns_64.mat', shuffle=True, subsample_time=5)
    n_total = data.shape[0]
    ntr = min(20000, int(0.8 * n_total))
    train_data = data[:ntr]
    ground_truth = train_data.squeeze(1).clone()
    spatial_dims = train_data.shape[2:]
    
    return {
        "train_data": train_data,
        "ground_truth": ground_truth[:1000],
        "spatial_dims": spatial_dims,
        "batch_size": 512,
        "batch_size_sig": 512,
        "modes": 16,
        "hch": 32,
        "pch": 64,
        "kernel_length": 0.001,
        "kernel_variance": 1.0,
        "epochs": 100,
        "n_gen_samples": 100,
        "is_2d": True,
    }


def setup_aemet():
    """Setup AEMET dataset."""
    import pandas as pd
    
    data_raw = pd.read_csv("../data/aemet.csv", index_col=0)
    data = torch.tensor(data_raw.values, dtype=torch.float32).squeeze().unsqueeze(1)
    
    min_data = torch.min(data)
    max_data = torch.max(data)
    rescaled_data = 6 * (data - min_data) / (max_data - min_data) - 3
    
    n_repeat = 50
    train_data = rescaled_data.repeat(n_repeat, 1, 1)
    ground_truth = rescaled_data.squeeze(1)
    n_x = 365
    
    return {
        "train_data": train_data,
        "ground_truth": ground_truth,
        "n_x": n_x,
        "batch_size": 512,
        "batch_size_sig": 64,
        "modes": 64,
        "width": 256,
        "mlp_width": 128,
        "kernel_length": 0.01,
        "kernel_variance": 0.1,
        "epochs": 300,
        "n_gen_samples": 500,
        "is_2d": False,
    }


def setup_expr_genes():
    """Setup Gene Expression dataset."""
    full_data = torch.load('../data/full_genes.pt').float()
    centered_loggen = full_data.log10() - full_data.log10().mean(1).unsqueeze(-1)
    expr_genes = centered_loggen[(centered_loggen.std(1) > .3), :]
    
    train_data = expr_genes.unsqueeze(1)
    ground_truth = expr_genes
    n_x = expr_genes.shape[1]
    
    return {
        "train_data": train_data,
        "ground_truth": ground_truth,
        "n_x": n_x,
        "batch_size": 512,
        "batch_size_sig": 128,
        "modes": 16,
        "width": 256,
        "mlp_width": 128,
        "kernel_length": 0.01,
        "kernel_variance": 0.1,
        "epochs": 300,
        "n_gen_samples": 500,
        "is_2d": False,
    }


def setup_economy():
    """Setup Economy datasets (averaged across 3 datasets)."""
    econ1 = torch.load('../data/economy/econ1.pt').float()
    econ2 = torch.load('../data/economy/econ2.pt').float()
    econ2 = econ2[~torch.any(econ2.isnan(), dim=1)]
    econ3 = torch.load('../data/economy/econ3.pt').float()
    
    econ1 = econ1 / torch.mean(econ1, dim=1).unsqueeze(-1)
    econ3 = econ3 / torch.mean(econ3, dim=1).unsqueeze(-1)
    
    def maxmin_rescale(data):
        dmax = torch.max(data)
        dmin = torch.min(data)
        return -1 + 2 * (data - dmin) / (dmax - dmin)
    
    econ1scaled = maxmin_rescale(econ1)
    econ2scaled = maxmin_rescale(econ2)
    econ3scaled = maxmin_rescale(econ3)
    
    n_repeat = 10
    
    return {
        "datasets": {
            "econ1_population": {
                "train_data": econ1scaled.repeat(n_repeat, 1).unsqueeze(1),
                "ground_truth": econ1scaled,
            },
            "econ2_gdp": {
                "train_data": econ2scaled.repeat(n_repeat, 1).unsqueeze(1),
                "ground_truth": econ2scaled,
            },
            "econ3_labor": {
                "train_data": econ3scaled.repeat(n_repeat, 1).unsqueeze(1),
                "ground_truth": econ3scaled,
            },
        },
        # n_x should be the number of time points
        # Get from the actual train_data shape (last dimension after processing)
        "n_x": econ1scaled.repeat(n_repeat, 1).unsqueeze(1).shape[-1],
        "batch_size": 512,
        "batch_size_sig": 128,
        "modes": 16,
        "width": 128,
        "mlp_width": 64,
        "kernel_length": 0.01,
        "kernel_variance": 0.1,
        "epochs": 300,
        "n_gen_samples": 500,
        "is_2d": False,
        "is_multi": True,
    }


def setup_heston():
    """Setup Heston dataset (standard, 100 steps)."""
    from data.Heston import generate_heston_dataset
    from pathlib import Path
    
    # Cache parameters
    n_samples = 5000
    n_steps = 100
    kappa = 1.0
    sigma = 0.3
    
    # Check for cached dataset
    cache_dir = Path('../data/cache/')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_filename = f"Heston_kappa{kappa}_sigma{sigma}_n{n_samples}_steps{n_steps}_seed42.pt"
    cache_path = cache_dir / cache_filename
    
    if cache_path.exists():
        print(f"Loading cached Heston dataset from {cache_path}...")
        dataset = torch.load(cache_path, weights_only=False)
    else:
        print(f"Generating Heston dataset ({n_steps} steps)...")
        dataset = generate_heston_dataset(
            n_samples=n_samples,
            n_steps=n_steps - 1,  # To get n_steps points total
            T=1.0,
            mu=0.05,
            kappa=kappa,
            theta=0.04,
            sigma=sigma,
            rho=-0.7,
            seed=42,
        )
        print(f"Saving dataset to cache: {cache_path}")
        torch.save(dataset, cache_path)

    train_data = dataset['log_V_normalized'].unsqueeze(1)
    ground_truth = dataset['log_V_normalized']
    n_x = train_data.shape[-1]

    return {
        "train_data": train_data,
        "ground_truth": ground_truth,
        "n_x": n_x,
        "batch_size": 512,
        "batch_size_sig": 128,
        "modes": 32,
        "width": 256,
        "mlp_width": 128,
        "kernel_length": 0.01,
        "kernel_variance": 0.1,
        "epochs": 300,
        "n_gen_samples": 500,
        "is_2d": False,
    }


def setup_heston_long():
    """Setup Heston dataset (long, 1000 steps)."""
    from data.Heston import generate_heston_long_dataset, HESTON_LONG_STEPS
    from pathlib import Path
    
    # Cache parameters
    n_samples = 5000
    n_steps = HESTON_LONG_STEPS
    kappa = 1.0
    sigma = 0.3
    
    # Check for cached dataset
    cache_dir = Path('../data/cache/')
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_filename = f"Heston_kappa{kappa}_sigma{sigma}_n{n_samples}_steps{n_steps}_seed42.pt"
    cache_path = cache_dir / cache_filename
    
    if cache_path.exists():
        print(f"Loading cached Heston-long dataset from {cache_path}...")
        dataset = torch.load(cache_path, weights_only=False)
    else:
        print(f"Generating Heston-long dataset ({n_steps} steps)...")
        dataset = generate_heston_long_dataset(
            n_samples=n_samples,
            n_steps=n_steps - 1,  # To get n_steps points total
            T=1.0,
            mu=0.05,
            kappa=kappa,
            theta=0.04,
            sigma=sigma,
            rho=-0.7,
            seed=42,
        )
        print(f"Saving dataset to cache: {cache_path}")
        torch.save(dataset, cache_path)

    train_data = dataset['log_V_normalized'].unsqueeze(1)
    ground_truth = dataset['log_V_normalized']
    n_x = train_data.shape[-1]

    return {
        "train_data": train_data,
        "ground_truth": ground_truth,
        "n_x": n_x,
        "batch_size": 256,  # Smaller batch for longer sequences
        "batch_size_sig": 64,  # Smaller batch for signature kernel with long sequences
        "modes": 64,  # More modes for longer sequences
        "width": 256,
        "mlp_width": 128,
        "kernel_length": 0.01,
        "kernel_variance": 0.1,
        "epochs": 300,
        "n_gen_samples": 500,
        "is_2d": False,
    }


SETUP_FUNCTIONS = {
    "kdv": setup_kdv,
    "navier_stokes": setup_navier_stokes,
    "navier_stokes_128": setup_navier_stokes_128,
    "navier_stokes_v1e5": setup_navier_stokes_v1e5,
    "stochastic_kdv": setup_stochastic_kdv,
    "stochastic_ns": setup_stochastic_ns,
    "aemet": setup_aemet,
    "expr_genes": setup_expr_genes,
    "economy": setup_economy,
    "heston": setup_heston,
    "heston-long": setup_heston_long,
}


# =============================================================================
# Training Functions
# =============================================================================

def create_model_1d(modes, width, mlp_width, device):
    """Create 1D FNO model."""
    from models.fno import FNO
    return FNO(
        modes,
        vis_channels=1,
        hidden_channels=width,
        proj_channels=mlp_width,
        x_dim=1,
        t_scaling=1000
    ).to(device)


def create_model_2d(modes, hch, pch, device):
    """Create 2D FNO model."""
    from models.fno import FNO
    return FNO(
        modes,
        vis_channels=1,
        hidden_channels=hch,
        proj_channels=pch,
        x_dim=2,
        t_scaling=1000.0
    ).to(device)


def build_ffm_kwargs(config: dict, setup: dict, device: str) -> dict:
    """Build kwargs for FFMModelOT from config dict."""
    kwargs = {
        "kernel_length": setup["kernel_length"],
        "kernel_variance": setup["kernel_variance"],
        "sigma_min": 1e-4,
        "device": device,
        "dtype": torch.float32,
        "use_ot": config.get("use_ot", False),
        "use_gp_prior": config.get("use_gp_prior", True),
    }

    if config.get("use_ot", False):
        if "ot_method" in config:
            kwargs["ot_method"] = config["ot_method"]
        if "ot_reg" in config:
            kwargs["ot_reg"] = config["ot_reg"]
        if "ot_kernel" in config:
            kwargs["ot_kernel"] = config["ot_kernel"]
        if "ot_coupling" in config:
            kwargs["ot_coupling"] = config["ot_coupling"]
        if "ot_kernel_params" in config:
            kwargs["ot_kernel_params"] = config["ot_kernel_params"]

    return kwargs


def apply_setup_overrides(setup: dict, config: dict) -> None:
    """Apply experiment-level model/prior overrides before training."""
    overrides = dict(config.get("setup_overrides", {}))

    legacy_keys = {
        "gp_kernel_length": "kernel_length",
        "gp_kernel_variance": "kernel_variance",
        "fno_modes": "modes",
    }
    for config_key, setup_key in legacy_keys.items():
        if config_key in config:
            overrides[setup_key] = config[config_key]

    if not overrides:
        return

    print("  Applying setup overrides:")
    for key, value in overrides.items():
        old_value = setup.get(key, "<missing>")
        setup[key] = value
        print(f"    {key}: {old_value} -> {value}")


def train_single_seed(
    dataset: str,
    kernel: str,
    config: dict,
    setup: dict,
    seed: int,
    save_dir: Path,
    device: str,
    train_data: torch.Tensor,
    ground_truth: torch.Tensor,
) -> Dict[str, Any]:
    """Train a single configuration with a single seed."""
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from functional_fm_ot import FFMModelOT
    from util.ot_monitoring import TrainingMonitor
    from util.eval import GenerationQualityMetrics, GenerationQualityMetrics2D
    
    print(f"\n  Training {dataset}/{kernel} (seed={seed})...")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create model
    is_2d = setup.get("is_2d", False)
    if is_2d:
        model = create_model_2d(
            setup["modes"], setup["hch"], setup["pch"], device
        )
    else:
        model = create_model_1d(
            setup["modes"], setup["width"], setup["mlp_width"], device
        )
    
    # Create FFM
    ffm_kwargs = build_ffm_kwargs(config, setup, device)
    ffm = FFMModelOT(model, **ffm_kwargs)
    
    # Create data loader. Signature-kernel variants need the smaller signature
    # batch size even when the method name is not literally "signature".
    uses_signature_kernel = config.get("ot_kernel") == "signature"
    batch_size = setup["batch_size_sig"] if uses_signature_kernel else setup["batch_size"]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Training
    monitor = TrainingMonitor(
        method_name=f"{dataset}_{kernel}",
        use_ot=config.get("use_ot", False),
        ot_kernel=config.get("ot_kernel", "") or "",
        track_batch_metrics=False,
    )
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    
    training_metrics = ffm.train(
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=setup["epochs"],
        scheduler=scheduler,
        eval_int=0,
        save_int=setup["epochs"],
        generate=False,
        save_path=save_dir,
        monitor=monitor,
    )
    
    # Generate samples
    n_gen = setup["n_gen_samples"]
    print(f"    Generating {n_gen} samples...")
    
    if is_2d:
        spatial_dims = setup["spatial_dims"]
        samples = ffm.sample(list(spatial_dims), n_samples=n_gen).cpu().squeeze()
    else:
        samples = ffm.sample([setup["n_x"]], n_samples=n_gen).cpu().squeeze()
    
    # Compute metrics - use 2D class for PDE datasets
    print(f"    Computing quality statistics ({'2D' if is_2d else '1D'})...")
    if is_2d:
        quality_metrics = GenerationQualityMetrics2D(
            config_name=f"{dataset}_{kernel}",
            ot_kernel=config.get("ot_kernel", "") or "",
            ot_method=config.get("ot_method", "") or "",
            ot_coupling=config.get("ot_coupling", "") or "",
            use_ot=config.get("use_ot", False),
        )
    else:
        quality_metrics = GenerationQualityMetrics(
            config_name=f"{dataset}_{kernel}",
            ot_kernel=config.get("ot_kernel", "") or "",
            ot_method=config.get("ot_method", "") or "",
            ot_coupling=config.get("ot_coupling", "") or "",
            use_ot=config.get("use_ot", False),
        )
    quality_metrics.compute_from_samples(ground_truth, samples)
    
    if training_metrics is not None:
        if training_metrics.train_losses:
            quality_metrics.final_train_loss = training_metrics.train_losses[-1]
            quality_metrics.set_convergence_metrics(training_metrics.train_losses)
        if training_metrics.epoch_times:
            quality_metrics.total_train_time = sum(training_metrics.epoch_times)
    
    # Save results
    torch.save(samples, save_dir / 'samples.pt')
    torch.save(model.state_dict(), save_dir / 'model.pt')
    if training_metrics is not None:
        training_metrics.save(save_dir / 'training_metrics.json')
    quality_metrics.save(save_dir / 'quality_metrics.json')
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return quality_metrics.to_dict()


def train_single_seed_diffusion(
    dataset: str,
    method: str,  # 'ddpm' or 'ncsn'
    setup: dict,
    seed: int,
    save_dir: Path,
    device: str,
    train_data: torch.Tensor,
    ground_truth: torch.Tensor,
) -> Dict[str, Any]:
    """Train a single diffusion model (DDPM/NCSN) with a single seed."""
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from diffusion import DiffusionModel
    from util.eval import GenerationQualityMetrics, GenerationQualityMetrics2D
    from util.util import make_grid
    
    print(f"\n  Training {dataset}/{method.upper()} (seed={seed})...")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Get diffusion config
    config = DIFFUSION_CONFIGS[method]
    
    # Create model
    is_2d = setup.get("is_2d", False)
    if is_2d:
        model = create_model_2d(
            setup["modes"], setup["hch"], setup["pch"], device
        )
    else:
        model = create_model_1d(
            setup["modes"], setup["width"], setup["mlp_width"], device
        )
    
    # Create DiffusionModel
    if config['method'] == 'DDPM':
        diffusion = DiffusionModel(
            model, 
            method=config['method'],
            T=config['T'],
            device=device,
            kernel_length=setup["kernel_length"],
            kernel_variance=setup["kernel_variance"],
            beta_min=config['beta_min'],
            beta_max=config['beta_max'],
            dtype=torch.float32,
        )
    else:  # NCSN
        diffusion = DiffusionModel(
            model, 
            method=config['method'],
            T=config['T'],
            device=device,
            kernel_length=setup["kernel_length"],
            kernel_variance=setup["kernel_variance"],
            sigma1=config['sigma1'],
            sigmaT=config['sigmaT'],
            precondition=config.get('precondition', True),
            dtype=torch.float32,
        )
    
    # Create data loader
    batch_size = setup["batch_size"]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50)
    
    import time
    start_time = time.time()
    
    training_metrics = diffusion.train(
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=setup["epochs"],
        scheduler=scheduler,
        eval_int=0,
        save_int=setup["epochs"],
        generate=False,
        save_path=save_dir,
    )
    
    train_time = time.time() - start_time
    
    # Generate samples
    n_gen = setup["n_gen_samples"]
    print(f"    Generating {n_gen} samples...")
    
    # Use train_dims from diffusion model (inferred during training) instead of setup["n_x"]
    # This is more reliable as it reflects the actual training data dimensions
    if is_2d:
        sample_dims = list(diffusion.train_dims)
    else:
        sample_dims = list(diffusion.train_dims)
    
    # Set train_support (grid) for sampling
    diffusion.train_support = make_grid(sample_dims).to(device)
    samples = diffusion.sample(
        dims=sample_dims,
        n_channels=1,
        n_samples=n_gen,
    ).cpu().squeeze()
    
    # Compute metrics - use 2D class for 2D spatial datasets
    print(f"    Computing quality statistics ({'2D' if is_2d else '1D'})...")
    if is_2d:
        quality_metrics = GenerationQualityMetrics2D(
            config_name=f"{dataset}_{method}",
            ot_kernel="",
            ot_method="",
            ot_coupling="",
            use_ot=False,
        )
    else:
        quality_metrics = GenerationQualityMetrics(
            config_name=f"{dataset}_{method}",
            ot_kernel="",
            ot_method="",
            ot_coupling="",
            use_ot=False,
        )
    quality_metrics.compute_from_samples(ground_truth, samples)
    
    # Set training time
    quality_metrics.total_train_time = train_time
    
    # Try to get loss from training_metrics if available
    if training_metrics is not None and hasattr(training_metrics, 'train_losses') and training_metrics.train_losses:
        quality_metrics.final_train_loss = training_metrics.train_losses[-1]
        quality_metrics.set_convergence_metrics(training_metrics.train_losses)
    
    # Save results
    torch.save(samples, save_dir / 'samples.pt')
    torch.save(model.state_dict(), save_dir / 'model.pt')
    if training_metrics is not None:
        training_metrics.save(save_dir / 'training_metrics.json')
    quality_metrics.save(save_dir / 'quality_metrics.json')
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return quality_metrics.to_dict()


def train_single_seed_gano(
    dataset: str,
    setup: dict,
    seed: int,
    save_dir: Path,
    device: str,
    train_data: torch.Tensor,
    ground_truth: torch.Tensor,
) -> Dict[str, Any]:
    """Train a single GANO model with a single seed (supports both 1D and 2D data)."""
    from torch.utils.data import DataLoader
    import torch.optim as optim
    from gano import GANO
    from util.eval import GenerationQualityMetrics, GenerationQualityMetrics2D
    
    is_2d = setup.get("is_2d", False)
    
    print(f"\n  Training {dataset}/GANO (seed={seed}, {'2D' if is_2d else '1D'})...")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Get GANO config
    config = GANO_CONFIG.copy()
    
    # Create Generator and Discriminator based on data dimensionality
    vis_channels = 1
    factor = config.get('factor', 0.5)  # Channel expansion factor
    
    if is_2d:
        from models.gano_models import Generator, Discriminator
        x_dim = 2  # 2D spatial
        in_channels = vis_channels + x_dim
        out_channels = vis_channels
        pad = config['pad']
        d_co_domain = config['d_co_domain']  # Reduced for 2D (memory)
        
        model_g = Generator(in_channels, out_channels, d_co_domain, pad=pad, factor=factor).to(device)
        model_d = Discriminator(in_channels, out_channels, d_co_domain, pad=pad, factor=factor).to(device)
    else:
        from models.gano_models import Generator1D, Discriminator1D
        x_dim = 1  # 1D spatial/temporal
        in_channels = vis_channels + x_dim
        out_channels = vis_channels
        pad = config.get('pad_1d', 0)
        d_co_domain = config.get('d_co_domain_1d', 32)  # Can be larger for 1D
        
        model_g = Generator1D(in_channels, out_channels, d_co_domain, pad=pad).to(device)
        model_d = Discriminator1D(in_channels, out_channels, d_co_domain, pad=pad).to(device)
    
    # Create GANO wrapper
    gano = GANO(
        model_d, model_g,
        l_grad=config['l_grad'],
        n_critic=config['n_critic'],
        kernel_length=setup["kernel_length"],
        kernel_variance=setup["kernel_variance"],
        device=device,
        dtype=torch.float32,
    )
    
    # Create data loader - use smaller batch size for 2D to save memory
    if is_2d:
        batch_size = min(setup["batch_size"], 64)  # Limit to 64 for 2D GANO
    else:
        batch_size = setup["batch_size"]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Training
    optimizer_g = optim.Adam(model_g.parameters(), lr=1e-3)
    optimizer_d = optim.Adam(model_d.parameters(), lr=1e-3)
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=25, gamma=0.1)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=25, gamma=0.1)
    
    import time
    start_time = time.time()
    
    # GANO doesn't return training metrics in the same way, so we manually track
    gano.train(
        train_loader=train_loader,
        D_optimizer=optimizer_d,
        G_optimizer=optimizer_g,
        epochs=setup["epochs"],
        D_scheduler=scheduler_d,
        G_scheduler=scheduler_g,
        eval_int=0,
        save_int=setup["epochs"],
        generate=False,
        save_path=save_dir,
    )
    
    train_time = time.time() - start_time
    
    # Generate samples
    n_gen = setup["n_gen_samples"]
    print(f"    Generating {n_gen} samples...")
    
    if is_2d:
        spatial_dims = setup["spatial_dims"]
        samples = gano.sample(
            dims=list(spatial_dims),
            n_channels=1,
            n_samples=n_gen,
        ).cpu().squeeze()
    else:
        n_x = setup["n_x"]
        samples = gano.sample(
            dims=[n_x],
            n_channels=1,
            n_samples=n_gen,
        ).cpu().squeeze()
    
    # Compute metrics
    print(f"    Computing quality statistics ({'2D' if is_2d else '1D'})...")
    if is_2d:
        quality_metrics = GenerationQualityMetrics2D(
            config_name=f"{dataset}_gano",
            ot_kernel="",
            ot_method="",
            ot_coupling="",
            use_ot=False,
        )
    else:
        quality_metrics = GenerationQualityMetrics(
            config_name=f"{dataset}_gano",
            ot_kernel="",
            ot_method="",
            ot_coupling="",
            use_ot=False,
        )
    quality_metrics.compute_from_samples(ground_truth, samples)
    
    # Set training time
    quality_metrics.total_train_time = train_time
    
    # Save results
    torch.save(samples, save_dir / 'samples.pt')
    torch.save(model_g.state_dict(), save_dir / 'model_G.pt')
    torch.save(model_d.state_dict(), save_dir / 'model_D.pt')
    quality_metrics.save(save_dir / 'quality_metrics.json')
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    return quality_metrics.to_dict()


def recompute_metrics_from_samples(
    dataset: str,
    kernel: str,
    seed_dir: Path,
    ground_truth: torch.Tensor,
) -> Dict[str, Any]:
    """Recompute quality metrics from saved samples.
    
    This is useful when metrics computation was updated but you don't want
    to re-run training.
    """
    from util.eval import GenerationQualityMetrics, GenerationQualityMetrics2D
    
    # Determine if this is a 2D spatial dataset (not all PDE datasets are 2D)
    is_2d = dataset in SPATIAL_2D_DATASETS
    
    samples_path = seed_dir / 'samples.pt'
    if not samples_path.exists():
        print(f"    No samples.pt found in {seed_dir}")
        return {}
    
    print(f"    Loading samples from {samples_path}...")
    samples = torch.load(samples_path, weights_only=False)
    
    # Load config if available
    config_path = seed_dir / 'config.json'
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    # Recompute metrics using appropriate class for 1D vs 2D
    print(f"    Recomputing quality metrics ({'2D' if is_2d else '1D'})...")
    
    if is_2d:
        quality_metrics = GenerationQualityMetrics2D(
            config_name=f"{dataset}_{kernel}",
            ot_kernel=config.get("ot_kernel", "") or "",
            ot_method=config.get("ot_method", "") or "",
            ot_coupling=config.get("ot_coupling", "") or "",
            use_ot=config.get("use_ot", False),
        )
    else:
        quality_metrics = GenerationQualityMetrics(
            config_name=f"{dataset}_{kernel}",
            ot_kernel=config.get("ot_kernel", "") or "",
            ot_method=config.get("ot_method", "") or "",
            ot_coupling=config.get("ot_kernel", "") or "",
            use_ot=config.get("use_ot", False),
        )
    quality_metrics.compute_from_samples(ground_truth, samples)
    
    # Load training metrics if available to preserve convergence info
    training_metrics_path = seed_dir / 'training_metrics.json'
    if training_metrics_path.exists():
        try:
            with open(training_metrics_path, 'r') as f:
                tm = json.load(f)
            if 'train_losses' in tm and tm['train_losses']:
                quality_metrics.final_train_loss = tm['train_losses'][-1]
                quality_metrics.set_convergence_metrics(tm['train_losses'])
            if 'epoch_times' in tm and tm['epoch_times']:
                quality_metrics.total_train_time = sum(tm['epoch_times'])
        except Exception as e:
            print(f"    Warning: Could not load training_metrics.json: {e}")
    
    # Save updated metrics
    quality_metrics.save(seed_dir / 'quality_metrics.json')
    print(f"    Saved updated quality_metrics.json")
    
    return quality_metrics.to_dict()


def run_experiment(
    dataset: str,
    kernel: str,
    seeds: List[int],
    output_dir: Path,
    device: str,
    base_outputs_dir: Path,
    load_existing: bool = False,
    recompute_metrics: bool = False,
    primary_metric: str = 'mean_mse',
    skip_existing: bool = False,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run experiment for dataset/kernel with multiple seeds.

    If load_existing=True, will try to load existing results from seed directories
    instead of re-running training.

    If recompute_metrics=True, will reload saved samples and recompute metrics.

    If skip_existing=True, will skip seeds that already have samples.pt saved.

    kernel can be: 'none', 'signature', 'rbf', 'euclidean', 'ddpm', 'ncsn', 'gano'

    primary_metric specifies which metric to optimize when selecting the best config.
    """
    
    # Validate dataset
    if dataset not in SETUP_FUNCTIONS:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Check if this is a diffusion method, GAN method, CFM-OT, CFM-indep, or CFM-kernel-OT
    is_diffusion = kernel in DIFFUSION_METHODS
    is_gano = kernel in GAN_METHODS
    is_cfm_ot = kernel in CFM_OT_METHODS
    is_cfm_indep = kernel in CFM_INDEP_METHODS
    is_cfm_kernel_ot = kernel in CFM_KERNEL_OT_METHODS
    is_baseline = kernel in BASELINE_METHODS

    # Validate kernel/method
    valid_options = (
        ['none', 'signature', 'rbf', 'euclidean']
        + BASELINE_METHODS
        + list(SIGNATURE_GEOMETRY_VARIANTS)
        + list(HESTON_SIGNATURE_SWEEP_CONFIGS)
    )
    if kernel not in valid_options:
        raise ValueError(f"Unknown kernel/method: {kernel}. Must be one of {valid_options}")

    # 2D datasets don't support signature kernel (including cfm_sig_ot)
    if (kernel in ('signature', 'cfm_sig_ot') or kernel in SIGNATURE_GEOMETRY_VARIANTS or kernel in HESTON_SIGNATURE_SWEEP_CONFIGS) and dataset in ['navier_stokes', 'stochastic_ns', 'navier_stokes_128']:
        raise ValueError(f"Signature kernel not supported for 2D dataset: {dataset}")
    if kernel in HESTON_SIGNATURE_SWEEP_CONFIGS and dataset != 'heston':
        raise ValueError(f"Heston signature sweep config {kernel} is intended for dataset=heston.")

    # For diffusion/GANO/CFM-OT models, use predefined configs
    if is_diffusion:
        config_name = kernel.upper()
        config = DIFFUSION_CONFIGS[kernel].copy()
        print(f"  Using {config_name} config: {config}")
    elif is_gano:
        config_name = "GANO"
        config = GANO_CONFIG.copy()
        print(f"  Using GANO config: {config}")
    elif is_cfm_ot:
        config_name = "CFM-OT(L2)"
        config = CFM_OT_CONFIG.copy()
        print(f"  Using CFM-OT(L2) config: {config}")
    elif is_cfm_indep:
        config_name = "CFM(indep)"
        config = CFM_INDEP_CONFIG.copy()
        print(f"  Using CFM(indep) config: {config}")
    elif kernel == 'cfm_rbf_ot':
        config_name = "CFM-OT(RBF)"
        config = CFM_RBF_OT_CONFIG.copy()
        # Apply dataset-specific sigma override
        if dataset in CFM_RBF_OT_SIGMA_OVERRIDES:
            override_sigma = CFM_RBF_OT_SIGMA_OVERRIDES[dataset]
            config["ot_kernel_params"] = {"sigma": override_sigma}
            print(f"  Using CFM-OT(RBF) config with dataset-specific sigma={override_sigma}: {config}")
        else:
            print(f"  Using CFM-OT(RBF) config: {config}")
    elif kernel in CFM_RBF_OT_SIGMA_VARIANTS:
        sigma = CFM_RBF_OT_SIGMA_VARIANTS[kernel]
        config_name = f"CFM-OT(RBF,s={sigma})"
        config = CFM_RBF_OT_CONFIG.copy()
        config["ot_kernel_params"] = {"sigma": sigma}
        print(f"  Using CFM-OT(RBF) sigma={sigma}: {config}")
    elif kernel in CFM_RBF_OT_REG_VARIANTS:
        reg = CFM_RBF_OT_REG_VARIANTS[kernel]
        # Use best sigma per dataset, vary reg
        sigma = CFM_RBF_OT_SIGMA_OVERRIDES.get(dataset, 5.0)
        config_name = f"CFM-OT(RBF,r={reg})"
        config = CFM_RBF_OT_CONFIG.copy()
        config["ot_reg"] = reg
        config["ot_kernel_params"] = {"sigma": sigma}
        print(f"  Using CFM-OT(RBF) sigma={sigma}, reg={reg}: {config}")
    elif kernel in CFM_RBF_OT_JOINT_VARIANTS:
        sigma, reg = CFM_RBF_OT_JOINT_VARIANTS[kernel]
        config_name = f"CFM-OT(RBF,s={sigma},r={reg})"
        config = CFM_RBF_OT_CONFIG.copy()
        config["ot_reg"] = reg
        config["ot_kernel_params"] = {"sigma": sigma}
        print(f"  Using CFM-OT(RBF) sigma={sigma}, reg={reg}: {config}")
    elif kernel == 'cfm_sig_ot':
        config_name = "CFM-OT(Sig)"
        config = CFM_SIG_OT_CONFIG.copy()
        print(f"  Using CFM-OT(Sig) config: {config}")
    elif kernel in SIGNATURE_GEOMETRY_VARIANTS:
        config_name = f"kFFM-Signature({kernel})"
        config = FALLBACK_CONFIGS["signature"].copy()
        params = FALLBACK_CONFIGS["signature"]["ot_kernel_params"].copy()
        params.update(SIGNATURE_GEOMETRY_VARIANTS[kernel])
        config["ot_kernel_params"] = params
        config["use_gp_prior"] = True
        print(f"  Using signature kFFM follow-up config: {config}")
    elif kernel in HESTON_SIGNATURE_SWEEP_CONFIGS:
        config_name = f"kFFM-HestonSignature({kernel})"
        config = deepcopy(HESTON_SIGNATURE_SWEEP_CONFIGS[kernel])
        print(f"  Using local Heston signature sweep config: {config}")
    elif kernel == 'euc_rbf' or kernel in KFFM_RBF_SIGMA_VARIANTS:
        sigma = KFFM_RBF_SIGMA_VARIANTS.get(kernel, 1.0)
        config_name = f"kFFM-EucRBF(s={sigma})"
        config = KFFM_EUC_RBF_CONFIG.copy()
        config["ot_kernel_params"] = {"sigma": sigma}
        print(f"  Using bounded Euclidean-RBF kFFM config: {config}")
    elif kernel == 'sobolev_rbf' or kernel in KFFM_SOB_RBF_SIGMA_VARIANTS or kernel in KFFM_SOB_RBF_PARAM_VARIANTS:
        if dataset not in PDE_DATASETS:
            raise ValueError(
                f"Sobolev RBF geometry is intended for PDE datasets; got {dataset}. "
                "Use euc_rbf/signature on path-space datasets."
            )
        if kernel in KFFM_SOB_RBF_PARAM_VARIANTS:
            sigma, sobolev_s = KFFM_SOB_RBF_PARAM_VARIANTS[kernel]
        else:
            sigma = KFFM_SOB_RBF_SIGMA_VARIANTS.get(kernel, 1.0)
            sobolev_s = 1.0
        config_name = f"kFFM-SobolevRBF(sigma={sigma},H{sobolev_s})"
        config = KFFM_SOB_RBF_CONFIG.copy()
        config["ot_kernel_params"] = {"sigma": sigma, "s": sobolev_s, "domain_length": 1.0}
        print(f"  Using Sobolev-RBF kFFM config: {config}")
    elif kernel in PDE_TUNED_GEOMETRY_CONFIGS:
        if dataset not in PDE_DATASETS:
            raise ValueError(
                f"Tuned PDE geometry methods are intended for PDE datasets; got {dataset}."
            )
        config_name = f"kFFM-tuned({kernel})"
        config = deepcopy(PDE_TUNED_GEOMETRY_CONFIGS[kernel])
        print(f"  Using tuned PDE geometry kFFM config: {config}")
    else:
        # Get best config (dynamically loaded from experiment results)
        config_name, config = get_best_config(dataset, kernel, base_outputs_dir, primary_metric=primary_metric)
    
    if config is None and not is_baseline:
        print(f"  Warning: No best config found for {dataset}/{kernel}, using fallback")
        config = FALLBACK_CONFIGS.get(kernel, FALLBACK_CONFIGS['none'])
        config_name = f"{kernel}_fallback"
    else:
        print(f"  Using best config: {config_name}")
        if '_source_metrics' in config:
            for k, v in config['_source_metrics'].items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.2e}")
    
    setup_fn = SETUP_FUNCTIONS[dataset]
    
    # Try to load existing results if requested
    if load_existing:
        print(f"\n{'=' * 70}")
        print(f"Loading existing results for {dataset.upper()}...")
        print(f"{'=' * 70}")
        
        all_metrics = []
        loaded_count = 0
        
        for seed in seeds:
            seed_dir = output_dir / f"seed_{seed}"
            metrics_file = seed_dir / "quality_metrics.json"
            
            if metrics_file.exists():
                try:
                    metrics = load_quality_metrics(metrics_file)
                    all_metrics.append(metrics)
                    loaded_count += 1
                    print(f"  Loaded seed {seed}: {metrics_file}")
                except Exception as e:
                    print(f"  Failed to load seed {seed}: {e}")
            else:
                print(f"  Missing: {metrics_file}")
        
        if loaded_count > 0:
            print(f"\n  Successfully loaded {loaded_count}/{len(seeds)} seeds")
            # Aggregate and return
            aggregated = aggregate_metrics(all_metrics, dataset)
            
            # Update summary
            summary = {
                "dataset": dataset,
                "kernel": kernel,
                "source_config_name": config.get('_source_config', 'loaded'),
                "source_metrics": config.get('_source_metrics', {}),
                "config": {k: v for k, v in config.items() if not k.startswith('_')},
                "seeds": seeds[:loaded_count],
                "n_seeds": loaded_count,
                "aggregated_metrics": aggregated,
                "individual_metrics": all_metrics,
                "loaded_from_existing": True,
            }
            
            with open(output_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            return aggregated
        else:
            print(f"  No existing results found, cannot proceed with --load-existing")
            return {}
    
    # Recompute metrics from saved samples if requested
    if recompute_metrics:
        print(f"\n{'=' * 70}")
        print(f"Recomputing metrics from saved samples for {dataset.upper()}...")
        print(f"{'=' * 70}")
        
        # Need to setup dataset to get ground truth
        setup = setup_fn()
        
        all_metrics = []
        recomputed_count = 0
        
        # Determine ground truth based on dataset type
        if setup.get("is_multi", False):
            # For multi-dataset, use the first one's ground truth
            first_sub = list(setup["datasets"].keys())[0]
            ground_truth = setup["datasets"][first_sub]["ground_truth"]
        else:
            ground_truth = setup["ground_truth"]
        
        for seed in seeds:
            seed_dir = output_dir / f"seed_{seed}"
            
            if (seed_dir / 'samples.pt').exists():
                try:
                    metrics = recompute_metrics_from_samples(
                        dataset=dataset,
                        kernel=kernel,
                        seed_dir=seed_dir,
                        ground_truth=ground_truth,
                    )
                    if metrics:
                        all_metrics.append(metrics)
                        recomputed_count += 1
                        print(f"  Recomputed seed {seed}")
                except Exception as e:
                    print(f"  Failed to recompute seed {seed}: {e}")
            else:
                print(f"  Missing samples.pt: {seed_dir}")
        
        if recomputed_count > 0:
            print(f"\n  Successfully recomputed {recomputed_count}/{len(seeds)} seeds")
            # Aggregate and return
            aggregated = aggregate_metrics(all_metrics, dataset)
            
            # Update summary
            summary = {
                "dataset": dataset,
                "kernel": kernel,
                "source_config_name": config.get('_source_config', 'recomputed'),
                "source_metrics": config.get('_source_metrics', {}),
                "config": {k: v for k, v in config.items() if not k.startswith('_')},
                "seeds": seeds[:recomputed_count],
                "n_seeds": recomputed_count,
                "aggregated_metrics": aggregated,
                "individual_metrics": all_metrics,
                "recomputed_metrics": True,
            }
            
            with open(output_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            return aggregated
        else:
            print(f"  No samples found, cannot proceed with --recompute-metrics")
            return {}
    
    print(f"\n{'=' * 70}")
    print(f"Setting up {dataset.upper()} dataset...")
    print(f"{'=' * 70}")
    
    setup = setup_fn()
    if batch_size is not None:
        # Route through setup_overrides so the override is recorded in each
        # run's saved config.json alongside any config-level overrides.
        config["setup_overrides"] = {
            **config.get("setup_overrides", {}),
            "batch_size": batch_size,
            "batch_size_sig": batch_size,
        }
    apply_setup_overrides(setup, config)

    # Handle multi-dataset case (economy)
    if setup.get("is_multi", False):
        all_metrics = []
        for sub_name, sub_data in setup["datasets"].items():
            print(f"\n  Sub-dataset: {sub_name}")
            for seed in seeds:
                seed_dir = output_dir / sub_name / f"seed_{seed}"
                seed_dir.mkdir(parents=True, exist_ok=True)

                if skip_existing and (seed_dir / 'samples.pt').exists():
                    print(f"    Skipping seed {seed} (samples.pt exists)")
                    metrics_file = seed_dir / 'quality_metrics.json'
                    if metrics_file.exists():
                        all_metrics.append(load_quality_metrics(metrics_file))
                    continue

                if is_diffusion:
                    metrics = train_single_seed_diffusion(
                        dataset=f"{dataset}/{sub_name}",
                        method=kernel,  # 'ddpm' or 'ncsn'
                        setup=setup,
                        seed=seed,
                        save_dir=seed_dir,
                        device=device,
                        train_data=sub_data["train_data"],
                        ground_truth=sub_data["ground_truth"],
                    )
                elif is_gano:
                    metrics = train_single_seed_gano(
                        dataset=f"{dataset}/{sub_name}",
                        setup=setup,
                        seed=seed,
                        save_dir=seed_dir,
                        device=device,
                        train_data=sub_data["train_data"],
                        ground_truth=sub_data["ground_truth"],
                    )
                else:
                    metrics = train_single_seed(
                        dataset=f"{dataset}/{sub_name}",
                        kernel=kernel,
                        config=config,
                        setup=setup,
                        seed=seed,
                        save_dir=seed_dir,
                        device=device,
                        train_data=sub_data["train_data"],
                        ground_truth=sub_data["ground_truth"],
                    )
                all_metrics.append(metrics)
    else:
        all_metrics = []
        for seed in seeds:
            seed_dir = output_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            if skip_existing and (seed_dir / 'samples.pt').exists():
                print(f"  Skipping seed {seed} (samples.pt exists)")
                metrics_file = seed_dir / 'quality_metrics.json'
                if metrics_file.exists():
                    all_metrics.append(load_quality_metrics(metrics_file))
                continue

            if is_diffusion:
                metrics = train_single_seed_diffusion(
                    dataset=dataset,
                    method=kernel,  # 'ddpm' or 'ncsn'
                    setup=setup,
                    seed=seed,
                    save_dir=seed_dir,
                    device=device,
                    train_data=setup["train_data"],
                    ground_truth=setup["ground_truth"],
                )
            elif is_gano:
                metrics = train_single_seed_gano(
                    dataset=dataset,
                    setup=setup,
                    seed=seed,
                    save_dir=seed_dir,
                    device=device,
                    train_data=setup["train_data"],
                    ground_truth=setup["ground_truth"],
                )
            else:
                metrics = train_single_seed(
                    dataset=dataset,
                    kernel=kernel,
                    config=config,
                    setup=setup,
                    seed=seed,
                    save_dir=seed_dir,
                    device=device,
                    train_data=setup["train_data"],
                    ground_truth=setup["ground_truth"],
                )
            all_metrics.append(metrics)
    
    # Aggregate metrics
    aggregated = aggregate_metrics(all_metrics, dataset)
    
    # Clean config for saving (remove internal keys)
    config_to_save = {k: v for k, v in config.items() if not k.startswith('_')}
    
    # Save aggregated results
    summary = {
        "dataset": dataset,
        "kernel": kernel,
        "source_config_name": config.get('_source_config', 'fallback'),
        "source_metrics": config.get('_source_metrics', {}),
        "config": config_to_save,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "aggregated_metrics": aggregated,
        "individual_metrics": all_metrics,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    return aggregated


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run seeded experiments for best OT configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--list', action='store_true',
                        help='List available datasets and kernels with best configs')
    parser.add_argument('--show-config', action='store_true',
                        help='Show the config that will be used (requires --dataset and --kernel)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset to run (e.g., kdv, aemet, heston)')
    parser.add_argument('--kernel', type=str, default=None,
                        help='Kernel type (none, signature, rbf, euclidean)')
    parser.add_argument('--metric', type=str, default='mean_mse',
                        choices=['mean_mse', 'variance_mse', 'autocorrelation_mse', 
                                 'spectrum_mse', 'spectrum_mse_log', 'skewness_mse', 
                                 'kurtosis_mse', 'seasonal_mse', 'mmd_rbf'],
                        help='Metric to optimize when selecting best config (default: mean_mse)')
    parser.add_argument('--n_seeds', type=int, default=10,
                        help='Number of seeds to run (default: 10)')
    parser.add_argument('--seed_start', type=int, default=0,
                        help='Starting seed index (default: 0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Run only this specific seed (for parallelization)')
    parser.add_argument('--output_dir', type=str, default='../outputs/seeded_runs',
                        help='Output directory for seeded runs (default: ../outputs/seeded_runs)')
    parser.add_argument('--base_outputs_dir', type=str, default='../outputs',
                        help='Base outputs directory to load experiment results from (default: ../outputs)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--load-existing', action='store_true',
                        help='Load existing results from seed directories instead of re-running')
    parser.add_argument('--recompute-metrics', action='store_true',
                        help='Reload saved samples and recompute metrics (useful after updating metric computation)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip seeds that already have samples.pt saved')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override the training batch size (and batch_size_sig). '
                             'Pair with a distinct --output_dir so sweep runs do not '
                             'overwrite the default cells')

    args = parser.parse_args()
    
    # Setup paths
    base_outputs_dir = Path(args.base_outputs_dir)
    
    if args.list:
        list_available_options(base_outputs_dir)
        return
    
    if args.show_config:
        if args.dataset is None or args.kernel is None:
            print("Error: --show-config requires --dataset and --kernel")
            return
        show_config(args.dataset, args.kernel, base_outputs_dir, primary_metric=args.metric)
        return
    
    if args.dataset is None or args.kernel is None:
        print("Error: --dataset and --kernel are required.")
        print("Use --list to see available options.")
        parser.print_help()
        return
    
    # Validate dataset
    valid_datasets = list(SETUP_FUNCTIONS.keys())
    if args.dataset not in valid_datasets:
        print(f"Error: Unknown dataset '{args.dataset}'")
        print(f"Available datasets: {valid_datasets}")
        return
    
    # Validate kernel/method
    valid_kernels = (
        ['none', 'signature', 'rbf', 'euclidean']
        + BASELINE_METHODS
        + list(SIGNATURE_GEOMETRY_VARIANTS)
        + list(HESTON_SIGNATURE_SWEEP_CONFIGS)
    )
    if args.kernel not in valid_kernels:
        print(f"Error: Unknown kernel/method '{args.kernel}'")
        print(f"Available options: {valid_kernels}")
        return
    
    # 2D datasets don't support signature kernel
    if (args.kernel == 'signature' or args.kernel in SIGNATURE_GEOMETRY_VARIANTS or args.kernel in HESTON_SIGNATURE_SWEEP_CONFIGS) and args.dataset in ['navier_stokes', 'stochastic_ns']:
        print(f"Error: Signature kernel not supported for 2D dataset: {args.dataset}")
        return
    if args.kernel in HESTON_SIGNATURE_SWEEP_CONFIGS and args.dataset != 'heston':
        print(f"Error: {args.kernel} is a Heston-only signature sweep config")
        return
    
    # Setup device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Disable cuDNN.  The FNO model uses spectral (FFT) convolutions and
    # kernel-size-1 Conv1d projections -- none of which benefit from cuDNN.
    # On slurm nodes the pip-bundled libcudnn.so is often not on
    # LD_LIBRARY_PATH, causing CUDNN_STATUS_NOT_INITIALIZED at runtime.
    # Disabling cuDNN avoids the error and keeps the CUDA context clean
    # for sigkernel's GPU kernels.
    if device == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.enabled = False
    
    # Setup seeds
    if args.seed is not None:
        seeds = [2**args.seed]
    else:
        seeds = get_seeds(args.n_seeds, args.seed_start)
    
    print(f"Seeds: {seeds}")
    
    # Setup output directory
    output_dir = Path(args.output_dir) / args.dataset / args.kernel
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Run experiment
    aggregated = run_experiment(
        dataset=args.dataset,
        kernel=args.kernel,
        seeds=seeds,
        output_dir=output_dir,
        device=device,
        base_outputs_dir=base_outputs_dir,
        load_existing=getattr(args, 'load_existing', False),
        recompute_metrics=getattr(args, 'recompute_metrics', False),
        primary_metric=args.metric,
        skip_existing=getattr(args, 'skip_existing', False),
        batch_size=args.batch_size,
    )
    
    # Print results
    print_results(aggregated, args.dataset, args.kernel)
    
    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
