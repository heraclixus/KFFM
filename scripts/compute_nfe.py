"""
Compute NFE (Number of Function Evaluations) for FFM and k-FFM across datasets.

This script measures the average number of neural network forward passes
required during ODE integration (sampling) for:
- FFM (independent): baseline flow matching without OT
- k-FFM: kernel-based FFM with OT pairing (various kernels)
- DDPM: diffusion model baseline (fixed timesteps)

=============================================================================
HOW NFE IS CALCULATED
=============================================================================

1. **DDPM (Diffusion Models)**:
   - Uses FIXED discrete timesteps: T = 1000 (typical)
   - Each timestep requires ONE model forward pass
   - **NFE = T = 1000** (always, regardless of data)

2. **NCSN (Noise Conditional Score Networks)**:
   - Uses T noise levels with M Langevin steps each
   - Typical: T=10, M=200 → **NFE = T × M = 2000**

3. **FFM / k-FFM (Flow Matching)**:
   - Uses ADAPTIVE ODE solver (dopri5 = Dormand-Prince 5(4))
   - Solver adjusts step size to meet specified tolerances
   - NFE varies based on:
     - **Tolerance values** (most important factor!)
       - Per FFM paper appendix:
         * 1D datasets: rtol=atol=1e-10 → ~600-700 NFE
         * 2D datasets: rtol=atol=1e-5  → ~60-150 NFE
     - Data dimensionality
     - Smoothness of learned vector field
     - ODE solver tolerances
   - **NFE ≈ 600-700** for AEMET (as reported in original FFM paper: 668)

=============================================================================
KEY INSIGHT
=============================================================================

For FFM/k-FFM, NFE depends on the TRAINED model's vector field smoothness.
- Smoother/straighter flows → fewer integration steps → lower NFE
- k-FFM with OT coupling produces straighter training paths
- This may (theoretically) result in slightly lower NFE after training

However, the NFE difference between FFM and k-FFM is typically small,
as both learn similar flows - the main benefit of k-FFM is faster/better
training convergence, not inference speed.

=============================================================================
Usage:
    python compute_nfe.py                    # Run all experiments
    python compute_nfe.py --n-samples 100    # Use fewer samples for quick test
    python compute_nfe.py --dataset aemet    # Only run on AEMET dataset
    python compute_nfe.py --load-models      # Load trained models (more accurate)
"""

import sys
sys.path.append('../')

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
from torchdiffeq import odeint
from functools import partial

from models.fno import FNO
from util.gaussian_process import GPPrior
from util.util import make_grid

# =============================================================================
# NFE Counter Wrapper
# =============================================================================

class NFECounter(nn.Module):
    """Wrapper to count function evaluations during ODE integration."""
    
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.nfe = 0
    
    def forward(self, t, x):
        self.nfe += 1
        return self.model(t, x)
    
    def reset(self):
        self.nfe = 0


def measure_nfe(
    model: nn.Module,
    gp: GPPrior,
    dims: Tuple[int, ...],
    n_channels: int = 1,
    n_samples: int = 100,
    batch_size: int = 50,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    device: str = 'cpu',
) -> Tuple[float, float]:
    """
    Measure average NFE during sampling.
    
    Parameters
    ----------
    model : nn.Module
        The neural network velocity field
    gp : GPPrior
        GP prior for base distribution
    dims : tuple
        Spatial dimensions (e.g., (365,) for AEMET)
    n_channels : int
        Number of channels
    n_samples : int
        Total samples to generate
    batch_size : int
        Batch size for generation
    rtol, atol : float
        ODE solver tolerances
    device : str
        
    Returns
    -------
    mean_nfe : float
        Mean NFE across all samples
    std_nfe : float
        Standard deviation of NFE
    """
    model = model.to(device)
    model.eval()
    
    # Wrap model with NFE counter
    nfe_model = NFECounter(model)
    
    nfe_list = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    t_eval = torch.linspace(0, 1, 2, device=device)
    grid = make_grid(dims)
    
    with torch.no_grad():
        for i in range(n_batches):
            current_batch_size = min(batch_size, n_samples - i * batch_size)
            if current_batch_size <= 0:
                break
            
            # Sample from GP prior
            x0 = gp.sample(grid, dims, n_samples=current_batch_size, n_channels=n_channels)
            x0 = x0.to(device)
            
            # Reset counter
            nfe_model.reset()
            
            # Integrate ODE
            _ = odeint(nfe_model, x0, t_eval, method='dopri5', rtol=rtol, atol=atol)
            
            # Record NFE (divide by batch size since it's counted per forward pass on the batch)
            nfe_list.append(nfe_model.nfe)
    
    return np.mean(nfe_list), np.std(nfe_list)


# =============================================================================
# Dataset Configurations
# =============================================================================

def get_dataset_configs() -> Dict:
    """
    Get configurations for all datasets.
    
    These match the configurations used in the experiment scripts.
    Note: output_dir must match the actual folder structure where models are saved.
    Note: modes must match what was used during training (from *_ot.py scripts)
    
    Per FFM paper appendix:
    - 1D datasets use rtol=atol=1e-10
    - 2D datasets use rtol=atol=1e-5
    """
    
    # Default tolerances per FFM paper
    TOL_1D = 1e-10  # 1D datasets: high quality
    TOL_2D = 1e-5   # 2D datasets: computational efficiency
    
    return {
        "AEMET": {
            "n_x": 365,
            "n_channels": 1,
            "modes": 64,  # from AEMET_ot.py
            "width": 256,
            "kernel_length": 0.01,
            "kernel_variance": 0.1,
            "output_dir": "AEMET_ot_comprehensive",
            "is_2d": False,
            "default_tol": TOL_1D,
        },
        "rBergomi": {
            "n_x": 100,
            "n_channels": 1,
            "modes": 32,  # from rBergomi_ot.py (non-long version)
            "width": 256,
            "kernel_length": 0.01,
            "kernel_variance": 0.1,
            "output_dir": "rBergomi_ot_H0p10",
            "is_2d": False,
            "default_tol": TOL_1D,
        },
        "Heston": {
            "n_x": 100,
            "n_channels": 1,
            "modes": 32,  # from Heston_ot.py (non-long version)
            "width": 256,
            "kernel_length": 0.01,
            "kernel_variance": 0.1,
            "output_dir": "Heston_ot_kappa1.0",
            "is_2d": False,
            "default_tol": TOL_1D,
        },
        # Econ has nested structure: econ_ot_comprehensive/econ1_population/...
        "Econ": {
            "n_x": 60,  # Years of data
            "n_channels": 1,
            "modes": 32,  # from econ_ot.py
            "width": 256,
            "kernel_length": 0.01,
            "kernel_variance": 0.1,
            "output_dir": "econ_ot_comprehensive/econ1_population",
            "is_2d": False,
            "default_tol": TOL_1D,
        },
        "Expr. Genes": {
            "n_x": 264,
            "n_channels": 1,
            "modes": 16,  # from expr_genes_ot.py
            "width": 256,
            "kernel_length": 0.01,
            "kernel_variance": 0.1,
            "output_dir": "expr_genes_ot_comprehensive",
            "is_2d": False,
            "default_tol": TOL_1D,
        },
        "KdV": {
            "n_x": 128,
            "n_channels": 1,
            "modes": 64,  # from kdv_ot.py
            "width": 256,
            "kernel_length": 0.01,
            "kernel_variance": 0.1,
            "output_dir": "kdv_ot",
            "is_2d": False,  # 1D PDE (spatial)
            "default_tol": TOL_1D,
        },
        "Stoch. KdV": {
            "n_x": 128,
            "n_channels": 1,
            "modes": 32,  # from stochastic_kdv_ot.py
            "width": 256,
            "kernel_length": 0.01,
            "kernel_variance": 0.1,
            "output_dir": "stochastic_kdv_ot",
            "is_2d": False,  # 1D PDE (spatial)
            "default_tol": TOL_1D,
        },
        # 2D PDE datasets would use TOL_2D
        # "Navier-Stokes": {..., "is_2d": True, "default_tol": TOL_2D},
    }


def load_trained_model(
    output_dir: str,
    config_name: str,
    seed: int,
    model_config: Dict,
    device: str,
) -> Optional[nn.Module]:
    """
    Try to load a trained model from the outputs directory.
    
    Returns None if model not found.
    """
    base_path = Path(f"../outputs/{output_dir}/{config_name}/seed_{seed}")
    
    if not base_path.exists():
        return None
    
    model_path = find_model_file(base_path)
    
    if model_path is not None:
        return _load_model_checkpoint(model_path, model_config, device)
    
    return None


def _load_model_checkpoint(model_path: Path, model_config: Dict, device: str) -> Optional[nn.Module]:
    """Load model from checkpoint, handling different save formats."""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different save formats
        if isinstance(checkpoint, nn.Module):
            # Saved as full model: torch.save(model, path)
            return checkpoint.to(device)
        elif isinstance(checkpoint, dict):
            model = create_model(model_config, device)
            # Clean up state_dict if needed
            state_dict = checkpoint
            if '_metadata' in state_dict:
                state_dict = {k: v for k, v in state_dict.items() if k != '_metadata'}
            # Remove 'module.' prefix if present (from DataParallel)
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            return model
        else:
            print(f"    Unknown checkpoint format: {type(checkpoint)}")
            return None
    except Exception as e:
        print(f"    Error loading checkpoint: {e}")
        return None


def find_model_file(base_path: Path) -> Optional[Path]:
    """Find model checkpoint file in a directory."""
    # Try different model file patterns
    patterns = [
        "model.pt",
        "epoch_*.pt",
        "*_model.pt",
        "checkpoint.pt",
    ]
    
    for pattern in patterns:
        files = list(base_path.glob(pattern))
        if files:
            # Return the latest (by name) if multiple matches
            return sorted(files)[-1]
    
    return None


def create_model(config: Dict, device: str) -> nn.Module:
    """Create FNO model from config."""
    return FNO(
        config["modes"],
        vis_channels=config["n_channels"],
        hidden_channels=config["width"],
        proj_channels=128,
        x_dim=1,
        t_scaling=1000
    ).to(device)


# =============================================================================
# Best Config Selection per Kernel Type
# =============================================================================

# Mapping of method types to config name patterns
# Separated into:
# 1. Independent (baseline FFM, no OT)
# 2. Gaussian OT (closed-form Bures-Wasserstein, NOT a kernel method)
# 3. Kernel OT methods (use discrete OT with kernel-induced RKHS costs)
METHOD_CONFIG_PATTERNS = {
    # Baseline
    "Independent": ["independent"],
    # Gaussian OT - parametric, closed-form (NOT kernel-based)
    "Gaussian OT": ["gaussian_ot"],
    # Kernel OT methods - non-parametric, discrete OT with kernel costs
    "Euclidean": ["euclidean_exact", "euclidean_sinkhorn"],
    "RBF": ["rbf_exact", "rbf_sinkhorn"],
    "Signature": ["signature_sinkhorn", "sig_"],
}

# For categorization in tables
GAUSSIAN_OT_METHODS = {"Gaussian OT"}
KERNEL_OT_METHODS = {"Euclidean", "RBF", "Signature"}

# Backward compatibility alias
KERNEL_CONFIG_PATTERNS = METHOD_CONFIG_PATTERNS


def find_best_config_per_kernel(output_dir: str, metric: str = "mean_mse") -> Dict[str, str]:
    """
    Find the best configuration for each kernel type based on quality metrics.
    
    Parameters
    ----------
    output_dir : str
        Path to output directory (e.g., "AEMET_ot_comprehensive")
    metric : str
        Metric to use for ranking (lower is better)
        
    Returns
    -------
    best_configs : dict
        Mapping from kernel type to best config name
    """
    import json
    
    base_path = Path(f"../outputs/{output_dir}")
    if not base_path.exists():
        print(f"  Warning: Output directory not found: {base_path}")
        return {}
    
    best_configs = {}
    
    for kernel_type, patterns in KERNEL_CONFIG_PATTERNS.items():
        best_score = float('inf')
        best_config = None
        
        # Search for configs matching this kernel type
        for config_dir in base_path.iterdir():
            if not config_dir.is_dir():
                continue
            
            config_name = config_dir.name
            
            # Check if this config matches the kernel pattern
            matches = any(p in config_name.lower() for p in patterns)
            if not matches:
                continue
            
            # Find quality metrics across seeds
            scores = []
            for seed_dir in config_dir.iterdir():
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                
                metrics_file = seed_dir / "quality_metrics.json"
                if metrics_file.exists():
                    with open(metrics_file) as f:
                        metrics = json.load(f)
                        if metric in metrics and metrics[metric] is not None:
                            scores.append(metrics[metric])
            
            if scores:
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_config = config_name
        
        if best_config:
            best_configs[kernel_type] = best_config
            print(f"  {kernel_type}: {best_config} ({metric}={best_score:.6f})")
    
    return best_configs


def load_best_model_for_kernel(
    output_dir: str,
    config_name: str,
    model_config: Dict,
    device: str,
    seed: int = 1,
) -> Optional[nn.Module]:
    """
    Load the trained model for a specific configuration.
    
    Returns None if no model checkpoint is found.
    """
    base_path = Path(f"../outputs/{output_dir}/{config_name}/seed_{seed}")
    
    if not base_path.exists():
        print(f"    ✗ Path not found: {base_path}")
        return None
    
    # Find model file
    model_path = find_model_file(base_path)
    
    if model_path is not None:
        print(f"    Loading model from: {model_path}")
        model = _load_model_checkpoint(model_path, model_config, device)
        if model is not None:
            print(f"    ✓ Model loaded successfully")
            return model
        else:
            print(f"    ✗ Failed to load model")
            return None
    else:
        # List what files ARE present
        files = list(base_path.glob("*"))
        file_names = [f.name for f in files if f.is_file()]
        print(f"    ✗ No model checkpoint found in {base_path}")
        print(f"      Files present: {file_names}")
        print(f"      (Need model.pt or epoch_*.pt for NFE measurement)")
    
    return None


# =============================================================================
# Main NFE Computation
# =============================================================================

def compute_all_nfe(
    n_samples: int = 100,
    batch_size: int = 50,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    datasets: Optional[List[str]] = None,
    device: str = None,
    load_trained: bool = True,
) -> pd.DataFrame:
    """
    Compute NFE for all datasets and kernel types.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate for NFE estimation
    batch_size : int
        Batch size for sampling
    rtol, atol : float, optional
        ODE solver tolerances. If None, uses per-dataset defaults from FFM paper:
        - 1D datasets: 1e-10
        - 2D datasets: 1e-5
    datasets : list, optional
        Specific datasets to run (None = all)
    device : str
        Device to use
    load_trained : bool
        If True, load trained models for each kernel type
        If False, use random initialization (same NFE for all)
        
    Returns
    -------
    results : pd.DataFrame
        Results with columns: Method, Kernel, Dataset, NFE, NFE_std
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Samples per dataset: {n_samples}")
    if rtol is None or atol is None:
        print(f"ODE tolerances: per-dataset (1e-10 for 1D, 1e-5 for 2D per FFM paper)")
    else:
        print(f"ODE tolerances: rtol={rtol}, atol={atol}")
    print(f"Load trained models: {load_trained}")
    print()
    
    all_configs = get_dataset_configs()
    
    # Filter datasets if specified
    if datasets:
        all_configs = {k: v for k, v in all_configs.items() 
                      if k.lower() in [d.lower() for d in datasets]}
    
    results = []
    
    for dataset_name, config in all_configs.items():
        # Determine tolerances for this dataset
        dataset_rtol = rtol if rtol is not None else config.get("default_tol", 1e-10)
        dataset_atol = atol if atol is not None else config.get("default_tol", 1e-10)
        is_2d = config.get("is_2d", False)
        
        print(f"{'='*60}")
        print(f"Dataset: {dataset_name} ({'2D' if is_2d else '1D'})")
        print(f"  n_x={config['n_x']}, modes={config['modes']}")
        print(f"  tolerances: rtol={dataset_rtol:.0e}, atol={dataset_atol:.0e}")
        print(f"{'='*60}")
        
        gp = GPPrior(
            lengthscale=config["kernel_length"],
            var=config["kernel_variance"],
            device=device
        )
        
        if load_trained:
            # Find best config for each kernel type
            print(f"\n  Finding best configs per kernel type...")
            output_dir = config.get("output_dir", "")
            best_configs = find_best_config_per_kernel(output_dir)
            
            if not best_configs:
                print(f"  No trained models found, using random initialization")
                load_trained = False
        
        if load_trained and best_configs:
            # Measure NFE for each kernel type's best model
            models_loaded = 0
            for kernel_type, config_name in best_configs.items():
                print(f"\n  Measuring NFE for {kernel_type} ({config_name})...")
                
                model = load_best_model_for_kernel(
                    output_dir=output_dir,
                    config_name=config_name,
                    model_config=config,
                    device=device,
                )
                
                if model is None:
                    print(f"    ⚠ Could not load model, using random init")
                    print(f"    ⚠ WARNING: Random init gives ~10-20 NFE, trained models give ~600-700 NFE!")
                    model = create_model(config, device)
                else:
                    models_loaded += 1
                
                mean_nfe, std_nfe = measure_nfe(
                    model=model,
                    gp=gp,
                    dims=(config["n_x"],),
                    n_channels=config["n_channels"],
                    n_samples=n_samples,
                    batch_size=batch_size,
                    rtol=dataset_rtol,
                    atol=dataset_atol,
                    device=device,
                )
                
                print(f"    NFE: {mean_nfe:.1f} ± {std_nfe:.1f}")
                
                # Determine method name
                if kernel_type == "Independent":
                    method = "FFM"
                    kernel = "-"
                elif kernel_type == "Gaussian OT":
                    method = "Gaussian OT"
                    kernel = "Gaussian OT"
                else:
                    method = "Kernel OT"
                    kernel = kernel_type
                
                results.append({
                    "Method": method,
                    "Kernel": kernel,
                    "Dataset": dataset_name,
                    "Config": config_name,
                    "NFE": mean_nfe,
                    "NFE_std": std_nfe,
                })
        else:
            # Use random model (same NFE for all kernel types)
            model = create_model(config, device)
            
            print(f"  Measuring NFE with random model...")
            mean_nfe, std_nfe = measure_nfe(
                model=model,
                gp=gp,
                dims=(config["n_x"],),
                n_channels=config["n_channels"],
                n_samples=n_samples,
                batch_size=batch_size,
                rtol=dataset_rtol,
                atol=dataset_atol,
                device=device,
            )
            
            print(f"  NFE: {mean_nfe:.1f} ± {std_nfe:.1f}")
            
            # Record same NFE for all methods
            for kernel_type in ["Independent", "Gaussian OT", "Euclidean", "RBF", "Signature"]:
                if kernel_type == "Independent":
                    method = "FFM"
                    kernel = "-"
                else:
                    method = "OT-FFM"
                    kernel = kernel_type
                
                results.append({
                    "Method": method,
                    "Kernel": kernel,
                    "Dataset": dataset_name,
                    "Config": "random_init",
                    "NFE": mean_nfe,
                    "NFE_std": std_nfe,
                })
    
    return pd.DataFrame(results)


def results_to_latex(df: pd.DataFrame, caption: str = None) -> str:
    """
    Convert results DataFrame to LaTeX table.
    
    Format: Kernel | Dataset | NFE (with DDPM comparison)
    """
    latex = r"""% =============================================================================
% NFE (Number of Function Evaluations) Comparison Table
% Generated by compute_nfe.py
% =============================================================================

"""
    
    # Main table: Kernel × Dataset → NFE
    latex += r"""\begin{table}[htbp]
\centering
"""
    if caption:
        latex += f"\\caption{{{caption}}}\n"
    else:
        latex += r"\caption{Number of Function Evaluations (NFE) for best configuration per kernel type}" + "\n"
    
    latex += r"""\label{tab:nfe_by_kernel}
\begin{tabular}{llr}
\toprule
\textbf{Kernel} & \textbf{Dataset} & \textbf{NFE} \\
\midrule
"""
    
    # Group by kernel and dataset
    for kernel in df['Kernel'].unique():
        kernel_df = df[df['Kernel'] == kernel]
        for _, row in kernel_df.iterrows():
            kernel_display = row['Kernel'] if row['Kernel'] != '-' else 'Independent'
            latex += f"{kernel_display} & {row['Dataset']} & {row['NFE']:.0f} \\\\\n"
        latex += r"\midrule" + "\n"
    
    # Remove last midrule and add bottomrule
    latex = latex.rsplit(r"\midrule", 1)[0]
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def generate_comparison_table(df: pd.DataFrame) -> str:
    """
    Generate comparison table: DDPM vs FFM vs Gaussian OT vs Kernel OT (best).
    
    Separates Gaussian OT (closed-form Bures-Wasserstein) from 
    Kernel OT (discrete OT with kernel costs).
    """
    latex = r"""
% =============================================================================
% NFE Comparison: DDPM vs FFM vs Gaussian OT vs Kernel OT
% =============================================================================
% Note: Gaussian OT uses closed-form Bures-Wasserstein solution (parametric)
%       Kernel OT uses discrete OT with RKHS kernel costs (non-parametric)

\begin{table}[htbp]
\centering
\caption{NFE comparison across methods}
\label{tab:nfe_comparison}
\begin{tabular}{lccccc}
\toprule
\textbf{Dataset} & \textbf{DDPM} & \textbf{FFM} & \textbf{Gaussian OT} & \textbf{Kernel OT} & \textbf{Best Kernel} \\
\midrule
"""
    
    ddpm_nfe = 1000
    
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        
        # Get FFM (Independent) NFE
        ffm_row = dataset_df[dataset_df['Kernel'] == '-']
        ffm_nfe = ffm_row['NFE'].values[0] if len(ffm_row) > 0 else None
        
        # Get Gaussian OT NFE
        gaussian_row = dataset_df[dataset_df['Kernel'] == 'Gaussian OT']
        gaussian_nfe = gaussian_row['NFE'].values[0] if len(gaussian_row) > 0 else None
        
        # Get best Kernel OT (lowest NFE among Euclidean, RBF, Signature)
        kernel_ot_df = dataset_df[dataset_df['Kernel'].isin(KERNEL_OT_METHODS)]
        if len(kernel_ot_df) > 0:
            best_idx = kernel_ot_df['NFE'].idxmin()
            best_kernel_nfe = kernel_ot_df.loc[best_idx, 'NFE']
            best_kernel = kernel_ot_df.loc[best_idx, 'Kernel']
        else:
            best_kernel_nfe = None
            best_kernel = "-"
        
        # Format row
        ffm_str = f"{ffm_nfe:.0f}" if ffm_nfe else "-"
        gaussian_str = f"{gaussian_nfe:.0f}" if gaussian_nfe else "-"
        kernel_str = f"{best_kernel_nfe:.0f}" if best_kernel_nfe else "-"
        
        latex += f"{dataset} & {ddpm_nfe} & {ffm_str} & {gaussian_str} & {kernel_str} & {best_kernel} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex


def generate_full_latex_table(df: pd.DataFrame) -> str:
    """
    Generate comprehensive LaTeX table with all methods.
    
    Separates methods into:
    - DDPM (baseline)
    - Independent (FFM baseline)
    - Gaussian OT (closed-form Bures-Wasserstein)
    - Kernel OT methods (Euclidean, RBF, Signature)
    """
    latex = r"""
% =============================================================================
% Full NFE Table by Method and Dataset
% =============================================================================
% Method categories:
% - DDPM: Fixed T=1000 timesteps
% - Independent: FFM with no OT coupling (baseline)
% - Gaussian OT: Closed-form Bures-Wasserstein OT (parametric, assumes Gaussian)
% - Kernel OT: Discrete OT with kernel-induced RKHS distances (non-parametric)

\begin{table}[htbp]
\centering
\caption{Complete NFE results for all methods and datasets}
\label{tab:nfe_full}
\begin{tabular}{llrr}
\toprule
\textbf{Method} & \textbf{Dataset} & \textbf{NFE} & \textbf{vs DDPM} \\
\midrule
\multicolumn{4}{l}{\textit{Baseline}} \\
"""
    
    ddpm_nfe = 1000
    
    # Add DDPM baseline for each dataset
    for dataset in df['Dataset'].unique():
        latex += f"DDPM & {dataset} & {ddpm_nfe} & - \\\\\n"
    
    latex += r"\midrule" + "\n"
    latex += r"\multicolumn{4}{l}{\textit{Flow Matching (Independent)}} \\" + "\n"
    
    # Add Independent (FFM)
    indep_df = df[df['Kernel'] == '-']
    for _, row in indep_df.iterrows():
        reduction = (1 - row['NFE'] / ddpm_nfe) * 100
        latex += f"FFM & {row['Dataset']} & {row['NFE']:.0f} & {reduction:.1f}\\% \\\\\n"
    
    latex += r"\midrule" + "\n"
    latex += r"\multicolumn{4}{l}{\textit{Gaussian OT (Bures-Wasserstein)}} \\" + "\n"
    
    # Add Gaussian OT
    gaussian_df = df[df['Kernel'] == 'Gaussian OT']
    if len(gaussian_df) > 0:
        for _, row in gaussian_df.iterrows():
            reduction = (1 - row['NFE'] / ddpm_nfe) * 100
            latex += f"Gaussian OT & {row['Dataset']} & {row['NFE']:.0f} & {reduction:.1f}\\% \\\\\n"
    else:
        latex += "\\multicolumn{4}{c}{(no results)} \\\\\n"
    
    latex += r"\midrule" + "\n"
    latex += r"\multicolumn{4}{l}{\textit{Kernel OT (RKHS distances)}} \\" + "\n"
    
    # Add Kernel OT methods (Euclidean, RBF, Signature)
    for kernel in ["Euclidean", "RBF", "Signature"]:
        kernel_df = df[df['Kernel'] == kernel]
        if len(kernel_df) > 0:
            for _, row in kernel_df.iterrows():
                reduction = (1 - row['NFE'] / ddpm_nfe) * 100
                latex += f"{kernel} & {row['Dataset']} & {row['NFE']:.0f} & {reduction:.1f}\\% \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}

% Notes:
% - DDPM uses fixed T=1000 timesteps
% - FFM/OT methods use adaptive dopri5 ODE solver
% - Per FFM paper appendix:
%   - 1D datasets: rtol=atol=1e-10 (expect ~600-700 NFE)
%   - 2D datasets: rtol=atol=1e-5  (expect ~60-150 NFE, computational efficiency)
% - Gaussian OT: Closed-form solution assuming Gaussian source/target distributions
% - Kernel OT: Discrete OT computed with kernel-induced RKHS distance costs
% - "vs DDPM" shows percentage reduction in function evaluations
"""
    
    return latex


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute NFE for FFM/k-FFM')
    parser.add_argument('--n-samples', type=int, default=100,
                        help='Number of samples for NFE estimation')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for sampling')
    parser.add_argument('--rtol', type=float, default=None,
                        help='ODE relative tolerance (default: 1e-10 for 1D, 1e-5 for 2D per FFM paper)')
    parser.add_argument('--atol', type=float, default=None,
                        help='ODE absolute tolerance (default: 1e-10 for 1D, 1e-5 for 2D per FFM paper)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to run (e.g., aemet, rbergomi)')
    parser.add_argument('--output', type=str, default='../outputs/nfe_results.tex',
                        help='Output path for LaTeX table')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--no-load', action='store_true',
                        help='Do not load trained models (use random init)')
    parser.add_argument('--full-table', action='store_true',
                        help='Generate full table with all kernel/dataset combinations')
    args = parser.parse_args()
    
    print("="*70)
    print("NFE (Number of Function Evaluations) Computation")
    print("="*70)
    print()
    print("Background:")
    print("-" * 70)
    print("• DDPM: Fixed NFE = T (typically 1000 timesteps)")
    print("• NCSN: Fixed NFE = T × M (typically 10 × 200 = 2000)")
    print("• FFM/k-FFM: Adaptive NFE via dopri5 ODE solver")
    print()
    print("This script measures NFE for trained models with different kernels:")
    print("  - Independent (baseline FFM)")
    print("  - Gaussian OT")
    print("  - Euclidean OT")
    print("  - RBF kernel OT")
    print("  - Signature kernel OT")
    print()
    if args.rtol is None and args.atol is None:
        print("ODE tolerances: using FFM paper defaults")
        print("  - 1D datasets: rtol=atol=1e-10 (expect ~600-700 NFE)")
        print("  - 2D datasets: rtol=atol=1e-5  (expect ~60-150 NFE)")
    else:
        rtol_val = args.rtol if args.rtol is not None else "default"
        atol_val = args.atol if args.atol is not None else "default"
        print(f"ODE tolerances: rtol={rtol_val}, atol={atol_val}")
    print("-" * 70)
    print()
    
    # Parse datasets
    datasets = None
    if args.dataset:
        datasets = [args.dataset]
    
    # Compute NFE
    results_df = compute_all_nfe(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        rtol=args.rtol,
        atol=args.atol,
        datasets=datasets,
        device=args.device,
        load_trained=not args.no_load,
    )
    
    print()
    print("="*70)
    print("Results Summary (by Kernel and Dataset)")
    print("="*70)
    print()
    
    # Show full results
    display_cols = ['Kernel', 'Dataset', 'NFE', 'NFE_std']
    if 'Config' in results_df.columns:
        display_cols.insert(2, 'Config')
    print(results_df[display_cols].to_string(index=False))
    print()
    
    # Show comparison with DDPM
    print("="*70)
    print("Reduction vs DDPM (T=1000)")
    print("="*70)
    results_df['vs_DDPM'] = ((1 - results_df['NFE'] / 1000) * 100).round(1).astype(str) + '%'
    print(results_df[['Kernel', 'Dataset', 'NFE', 'vs_DDPM']].to_string(index=False))
    print()
    
    # Generate LaTeX
    latex_output = results_to_latex(results_df)
    latex_output += generate_comparison_table(results_df)
    
    if args.full_table:
        latex_output += generate_full_latex_table(results_df)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(latex_output)
    
    print(f"LaTeX table saved to: {output_path}")
    print()
    print("="*70)
    print("LaTeX Output:")
    print("="*70)
    print(latex_output)
    
    # Also save CSV for convenience
    csv_path = output_path.with_suffix('.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")
    
    # Print reference values from original FFM paper
    print()
    print("="*70)
    print("Reference: Original FFM Paper (Kerrigan et al.)")
    print("="*70)
    print("• AEMET dataset (FFM paper values):")
    print("  - DDPM: NFE = 1000 (fixed timesteps)")
    print("  - FFM:  NFE = 668 (adaptive dopri5, rtol=atol=1e-10)")
    print("  - Reduction: 33.2%")
    print()
    print("• FFM paper tolerance settings (Appendix):")
    print("  - 1D datasets: rtol=atol=1e-10 → ~600-700 NFE")
    print("  - 2D datasets: rtol=atol=1e-5  → ~60-150 NFE (computational efficiency)")
    print()