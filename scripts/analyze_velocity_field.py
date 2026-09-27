#!/usr/bin/env python3
"""
Analyze velocity field norm distributions for flow matching models.

This script implements two analysis approaches:
1. Velocity norm along trajectories - tracking ||v_θ(t, x_t)|| during ODE integration
2. Velocity norm distribution at time slices - histograms at fixed t values

These analyses help understand sampling stability beyond NFE metrics by examining:
- Maximum velocity norms (potential numerical instability)
- Velocity norm variance over time (flow straightness/uniformity)
- Distribution characteristics at different flow stages

Usage:
    # ===== RECOMMENDED: Load trained model checkpoints =====
    
    # Single checkpoint
    python analyze_velocity_field.py --checkpoint path/to/model.pt --dataset aemet
    
    # Multiple checkpoints (compare different methods)
    python analyze_velocity_field.py --checkpoint none=path/to/ffm.pt signature=path/to/sig.pt --dataset aemet
    
    # Scan a directory for all seed_*/model.pt files
    python analyze_velocity_field.py --scan-dir ../outputs/seeded_runs/aemet --dataset aemet
    
    # ===== Alternative: Train fresh models =====
    
    # Quick analysis with random models (for testing)
    python analyze_velocity_field.py --no-train --n-samples 50
    
    # Full analysis training models
    python analyze_velocity_field.py --dataset aemet --epochs 100
    
    # Analyze specific kernels
    python analyze_velocity_field.py --kernels none signature rbf
"""

import sys
sys.path.append('../')

import argparse
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from torchdiffeq import odeint
from collections import defaultdict
import warnings

from models.fno import FNO
from functional_fm_ot import FFMModelOT
from util.gaussian_process import GPPrior
from util.util import make_grid

# Suppress matplotlib font warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class VelocityNormStats:
    """Statistics for velocity norms."""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    q95: float  # 95th percentile
    q99: float  # 99th percentile
    
    @classmethod
    def from_tensor(cls, norms: torch.Tensor) -> 'VelocityNormStats':
        """Compute statistics from a tensor of norms."""
        norms_np = norms.detach().cpu().numpy().flatten()
        return cls(
            mean=float(np.mean(norms_np)),
            std=float(np.std(norms_np)),
            min=float(np.min(norms_np)),
            max=float(np.max(norms_np)),
            median=float(np.median(norms_np)),
            q25=float(np.percentile(norms_np, 25)),
            q75=float(np.percentile(norms_np, 75)),
            q95=float(np.percentile(norms_np, 95)),
            q99=float(np.percentile(norms_np, 99)),
        )


@dataclass
class TrajectoryAnalysis:
    """Results from velocity norm analysis along trajectories."""
    time_points: List[float]
    stats_per_time: List[VelocityNormStats]
    # Aggregated statistics
    overall_max: float = 0.0
    overall_mean: float = 0.0
    time_variance: float = 0.0  # Variance of mean norms over time (flow uniformity)
    # Integration steps (NFE)
    mean_nfe: float = 0.0
    std_nfe: float = 0.0
    
    def compute_aggregates(self):
        """Compute aggregate statistics."""
        means = [s.mean for s in self.stats_per_time]
        maxes = [s.max for s in self.stats_per_time]
        self.overall_max = max(maxes) if maxes else 0.0
        self.overall_mean = np.mean(means) if means else 0.0
        self.time_variance = np.var(means) if means else 0.0


@dataclass
class AggregatedResults:
    """Aggregated results from multiple runs."""
    method_name: str
    n_runs: int
    # Flow uniformity (time variance) statistics across runs
    uniformity_mean: float = 0.0
    uniformity_std: float = 0.0
    # NFE statistics across runs
    nfe_mean: float = 0.0
    nfe_std: float = 0.0
    # Individual run values (for detailed analysis)
    uniformity_values: List[float] = field(default_factory=list)
    nfe_values: List[float] = field(default_factory=list)
    
    @classmethod
    def from_trajectory_analyses(cls, method_name: str, analyses: List[TrajectoryAnalysis]) -> 'AggregatedResults':
        """Aggregate statistics from multiple trajectory analyses."""
        n_runs = len(analyses)
        uniformity_values = [a.time_variance for a in analyses]
        nfe_values = [a.mean_nfe for a in analyses]
        
        return cls(
            method_name=method_name,
            n_runs=n_runs,
            uniformity_mean=float(np.mean(uniformity_values)),
            uniformity_std=float(np.std(uniformity_values)),
            nfe_mean=float(np.mean(nfe_values)),
            nfe_std=float(np.std(nfe_values)),
            uniformity_values=uniformity_values,
            nfe_values=nfe_values,
        )


@dataclass
class DistributionAnalysis:
    """Results from velocity norm distribution analysis at time slices."""
    time_points: List[float]
    histograms: Dict[float, Tuple[np.ndarray, np.ndarray]]  # t -> (counts, bin_edges)
    stats_per_time: Dict[float, VelocityNormStats]


@dataclass
class MethodResults:
    """Complete analysis results for a method (kernel configuration)."""
    method_name: str
    trajectory_analysis: TrajectoryAnalysis
    distribution_analysis: DistributionAnalysis
    config: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Dataset Configurations
# =============================================================================

DATASET_CONFIGS = {
    'aemet': {
        'spatial_dims': (365,),
        'n_channels': 1,
        'modes': 64,
        'width': 256,
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        't_scaling': 1000,
        'data_file': '../data/aemet.csv',
    },
    'rbergomi': {
        'spatial_dims': (64,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        't_scaling': 1000,
        'data_file': '../data/rBergomi_H0p10_n5000.pt',
    },
    'heston': {
        'spatial_dims': (64,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        't_scaling': 1000,
        'data_file': '../data/Heston_kappa1.0_sigma0.3_n5000.pt',
    },
    'kdv': {
        'spatial_dims': (512,),
        'n_channels': 1,
        'modes': 64,
        'width': 256,
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        't_scaling': 1000,
        'data_file': '../data/KdV.mat',
    },
    'stochastic_kdv': {
        'spatial_dims': (128,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        't_scaling': 1000,
        'data_file': '../data/stochastic_kdv.mat',
    },
    'navier_stokes': {
        'spatial_dims': (64, 64),
        'n_channels': 1,
        'modes': 16,
        'width': 32,
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        't_scaling': 1000,
        'data_file': '../data/ns.mat',
        'is_2d': True,
    },
    'stochastic_ns': {
        'spatial_dims': (64, 64),
        'n_channels': 1,
        'modes': 16,
        'width': 32,
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        't_scaling': 1000,
        'data_file': '../data/stochastic_ns_64.mat',
        'is_2d': True,
    },
    'economy': {
        'spatial_dims': (128,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        't_scaling': 1000,
        'data_file': '../data/economy/econ1.pt',
    },
    'expr_genes': {
        'spatial_dims': (64,),
        'n_channels': 1,
        'modes': 16,
        'width': 256,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        't_scaling': 1000,
        'data_file': '../data/expr_genes.pt',
    },
}

# Kernel configurations for k-FFM
KERNEL_CONFIGS = {
    'none': {
        'use_ot': False,
        'description': 'Independent (no OT)',
    },
    'euclidean': {
        'use_ot': True,
        'ot_method': 'sinkhorn',
        'ot_kernel': 'euclidean',
        'ot_reg': 0.1,
        'description': 'Euclidean OT',
    },
    'rbf': {
        'use_ot': True,
        'ot_method': 'sinkhorn',
        'ot_kernel': 'rbf',
        'ot_kernel_params': {'sigma': 1.0},
        'ot_reg': 0.1,
        'description': 'RBF Kernel OT',
    },
    'signature': {
        'use_ot': True,
        'ot_method': 'sinkhorn',
        'ot_kernel': 'signature',
        'ot_kernel_params': {
            'order': 2,
            'sigma': 1.0,
            'normalize': True,
            'time_aug': True,
        },
        'ot_reg': 0.1,
        'description': 'Signature Kernel OT',
    },
    'gaussian_ot': {
        'use_ot': True,
        'ot_method': 'gaussian',
        'ot_reg': 0.01,
        'description': 'Gaussian OT (Bures-Wasserstein)',
    },
}


# =============================================================================
# Model Loading / Creation
# =============================================================================

def create_model(config: Dict, device: str) -> nn.Module:
    """Create FNO model from config."""
    is_2d = config.get('is_2d', False)
    x_dim = 2 if is_2d else 1
    
    return FNO(
        modes=config['modes'],
        vis_channels=config['n_channels'],
        hidden_channels=config['width'],
        proj_channels=config.get('proj_channels', 128),
        x_dim=x_dim,
        t_scaling=config.get('t_scaling', 1000),
    ).to(device)


def infer_model_config_from_checkpoint(
    checkpoint_path: Path,
    base_config: Dict,
    device: str = 'cpu',
) -> Dict:
    """Infer model configuration from checkpoint or its config.json.
    
    Attempts to determine model architecture by:
    1. Loading config.json from checkpoint directory
    2. Inferring from state_dict tensor shapes
    3. Falling back to base_config
    
    Returns updated config dict.
    """
    checkpoint_path = Path(checkpoint_path)
    config = base_config.copy()
    
    # Always try to infer from state_dict first (most reliable)
    state_dict = None
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove module. prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    except Exception as e:
        print(f"    Warning: Could not load checkpoint for inference: {e}")
    
    # Try to load config.json from same directory or parent directories
    config_paths = [
        checkpoint_path.parent / 'config.json',
        checkpoint_path.parent.parent / 'config.json',  # e.g., seed_1/../config.json
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    saved_config = json.load(f)
                
                # Extract relevant model parameters
                if 'fno_modes' in saved_config:
                    config['modes'] = saved_config['fno_modes']
                elif 'modes' in saved_config:
                    config['modes'] = saved_config['modes']
                
                if 'fno_width' in saved_config:
                    config['width'] = saved_config['fno_width']
                elif 'hidden_channels' in saved_config:
                    config['width'] = saved_config['hidden_channels']
                elif 'width' in saved_config:
                    config['width'] = saved_config['width']
                
                if 'proj_channels' in saved_config:
                    config['proj_channels'] = saved_config['proj_channels']
                elif 'fno_proj_channels' in saved_config:
                    config['proj_channels'] = saved_config['fno_proj_channels']
                
                if 'n_x' in saved_config:
                    # Infer spatial dims
                    n_x = saved_config['n_x']
                    if isinstance(n_x, int):
                        config['spatial_dims'] = (n_x,)
                
                print(f"    Loaded config from {config_path}: modes={config.get('modes')}, width={config.get('width')}, proj_channels={config.get('proj_channels', 128)}")
                break
            except Exception as e:
                print(f"    Warning: Could not parse {config_path}: {e}")
    
    # Now verify/override with state_dict inference (most reliable source of truth)
    if state_dict is not None:
        inferred = _infer_architecture_from_state_dict(state_dict)
        
        # Only override if we got valid values from state_dict
        if inferred.get('width', 0) > 4:  # Sanity check
            config['width'] = inferred['width']
        if inferred.get('modes', 0) > 0:
            config['modes'] = inferred['modes']
        if inferred.get('proj_channels', 0) > 0:
            config['proj_channels'] = inferred['proj_channels']
        
        print(f"    Final config from state_dict: modes={config.get('modes')}, width={config.get('width')}, proj_channels={config.get('proj_channels', 128)}")
    
    return config


def _infer_architecture_from_state_dict(state_dict: Dict) -> Dict:
    """Infer FNO architecture from state_dict tensor shapes.
    
    Note: neuralop's FNO stores Fourier modes differently. If you pass modes=M to
    the constructor, it creates weights with shape [..., M//2 + 1] for 1D.
    So we need to convert: tensor_shape -> constructor_modes = (tensor_shape - 1) * 2
    """
    result = {}
    
    # Remove module. prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Method 1: From FNO blocks (most direct)
    for key, tensor in state_dict.items():
        if 'fno_blocks' in key and 'weight' in key and 'tensor' in key:
            if tensor.dim() == 3:
                result['width'] = tensor.shape[0]
                # neuralop stores M//2 + 1 modes, so convert back: modes = (K-1)*2
                tensor_modes = tensor.shape[2]
                result['modes'] = (tensor_modes - 1) * 2
                print(f"    State dict - FNO blocks: width={result['width']}, tensor_modes={tensor_modes}, constructor_modes={result['modes']}")
            elif tensor.dim() == 4:
                result['width'] = tensor.shape[0]
                # Same conversion for 2D
                tensor_modes = tensor.shape[2]
                result['modes'] = (tensor_modes - 1) * 2
                print(f"    State dict - FNO blocks (2D): width={result['width']}, tensor_modes={tensor_modes}, constructor_modes={result['modes']}")
            break
    
    # Method 2: From lifting layer (backup)
    if 'width' not in result or result.get('width', 0) < 4:
        for key, tensor in state_dict.items():
            if 'lifting.fcs.0.weight' in key:
                # Shape: [hidden_channels, in_channels, 1]
                result['width'] = tensor.shape[0]
                print(f"    State dict - lifting.fcs.0: width={result['width']}")
                break
    
    # Method 3: From projection layer input
    for key, tensor in state_dict.items():
        if 'projection.fcs.0.weight' in key:
            # Shape: [proj_channels, hidden_channels, 1]
            result['proj_channels'] = tensor.shape[0]
            # The input dimension should be the FNO hidden_channels
            proj_input_dim = tensor.shape[1]
            if 'width' not in result or result.get('width', 0) < 4:
                result['width'] = proj_input_dim
                print(f"    State dict - projection input: width={result['width']}")
            print(f"    State dict - projection: proj_channels={result['proj_channels']}")
            break
    
    return result


def load_model_checkpoint(
    model_path: Path,
    config: Dict,
    device: str,
    verbose: bool = True,
) -> Optional[nn.Module]:
    """Load model from checkpoint file.
    
    Handles various checkpoint formats:
    - Full model saved with torch.save(model, path)
    - State dict saved with torch.save(model.state_dict(), path)
    - State dict with 'model_state_dict' key
    - DataParallel models with 'module.' prefix
    
    Automatically infers model architecture from checkpoint if possible.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        if verbose:
            print(f"  ✗ Checkpoint not found: {model_path}")
        return None
    
    try:
        # First, try to infer the correct model config from checkpoint
        inferred_config = infer_model_config_from_checkpoint(model_path, config, device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Case 1: Full model object
        if isinstance(checkpoint, nn.Module):
            if verbose:
                print(f"  ✓ Loaded full model from {model_path}")
            return checkpoint.to(device)
        
        # Case 2: Dictionary (state_dict or nested)
        elif isinstance(checkpoint, dict):
            model = create_model(inferred_config, device)
            
            # Check for nested state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                # Could be a full model inside dict
                if isinstance(checkpoint['model'], nn.Module):
                    if verbose:
                        print(f"  ✓ Loaded model from checkpoint dict")
                    return checkpoint['model'].to(device)
                state_dict = checkpoint['model']
            else:
                # Assume the dict itself is the state_dict
                state_dict = checkpoint
            
            # Remove 'module.' prefix from DataParallel
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k[7:] if k.startswith('module.') else k: v 
                             for k, v in state_dict.items()}
            
            # Remove metadata if present
            state_dict = {k: v for k, v in state_dict.items() 
                         if not k.startswith('_')}
            
            try:
                model.load_state_dict(state_dict)
            except RuntimeError as e:
                if 'size mismatch' in str(e):
                    if verbose:
                        print(f"  ⚠ Shape mismatch, attempting to re-infer config...")
                    # Try harder to infer the right shape
                    inferred_config = infer_model_config_from_state_dict(state_dict, config)
                    model = create_model(inferred_config, device)
                    model.load_state_dict(state_dict)
                else:
                    raise
            
            if verbose:
                print(f"  ✓ Loaded state dict from {model_path}")
            return model
        
        else:
            if verbose:
                print(f"  ✗ Unknown checkpoint format: {type(checkpoint)}")
            return None
            
    except Exception as e:
        if verbose:
            print(f"  ✗ Error loading {model_path}: {e}")
        return None


def infer_model_config_from_state_dict(state_dict: Dict, base_config: Dict) -> Dict:
    """Infer model config directly from state_dict tensor shapes.
    
    Note: neuralop's FNO stores Fourier modes differently. If you pass modes=M to
    the constructor, it creates weights with shape [..., M//2 + 1] for 1D.
    So we need to convert: tensor_shape -> constructor_modes = (tensor_shape - 1) * 2
    """
    config = base_config.copy()
    
    # Remove module. prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # First, try to infer from FNO blocks (most reliable)
    for key, tensor in state_dict.items():
        # FNO blocks weight: [hidden, hidden, modes] or similar
        if 'fno_blocks' in key and 'weight' in key and 'tensor' in key:
            if tensor.dim() == 3:
                config['width'] = tensor.shape[0]
                # neuralop stores M//2 + 1 modes, so convert back: modes = (K-1)*2
                tensor_modes = tensor.shape[2]
                config['modes'] = (tensor_modes - 1) * 2
                print(f"    From fno_blocks: width={config['width']}, tensor_modes={tensor_modes}, constructor_modes={config['modes']}")
                break
            elif tensor.dim() == 4:
                # 2D case: [hidden, hidden, modes_x, modes_y]
                config['width'] = tensor.shape[0]
                tensor_modes = tensor.shape[2]
                config['modes'] = (tensor_modes - 1) * 2
                print(f"    From fno_blocks (2D): width={config['width']}, tensor_modes={tensor_modes}, constructor_modes={config['modes']}")
                break
    
    # If FNO blocks didn't give us width, try lifting layer
    if 'width' not in config or config.get('width', 0) < 4:
        for key, tensor in state_dict.items():
            # lifting.fcs.0.weight shape: [hidden_channels, in_channels, 1]
            if 'lifting.fcs.0.weight' in key:
                config['width'] = tensor.shape[0]
                print(f"    From lifting layer: width={config['width']}")
                break
    
    # Infer proj_channels from projection layer
    for key, tensor in state_dict.items():
        if 'projection.fcs.0.weight' in key:
            # Shape: [proj_channels, hidden_channels, 1] or [proj_channels, hidden_channels, 1, 1]
            config['proj_channels'] = tensor.shape[0]
            # Also verify/update width from projection input dimension
            if tensor.shape[1] != config.get('width', 0) and tensor.shape[1] > 1:
                # The projection input is the FNO output width
                if 'width' not in config or config['width'] < 4:
                    config['width'] = tensor.shape[1]
                    print(f"    Updated width from projection input: {config['width']}")
            break
    
    # Also check lifting.fcs.1.weight for intermediate dimensions
    # Shape: [proj_channels_lifting, hidden_channels, 1]
    for key, tensor in state_dict.items():
        if 'lifting.fcs.1.weight' in key:
            # This gives us the lifting output dimension which should match FNO input
            lifting_out = tensor.shape[0]
            lifting_in = tensor.shape[1]
            print(f"    Lifting layer: {lifting_in} -> {lifting_out}")
            # If width wasn't set properly, use lifting_in (hidden_channels)
            if config.get('width', 0) < 4:
                config['width'] = lifting_in
            break
    
    print(f"    Re-inferred from state_dict: modes={config.get('modes')}, width={config.get('width')}, proj_channels={config.get('proj_channels', 128)}")
    
    return config


def find_model_file(base_path: Path) -> Optional[Path]:
    """Find model checkpoint file in directory."""
    patterns = ["model.pt", "epoch_*.pt", "*_model.pt", "checkpoint.pt", "*.pt"]
    
    for pattern in patterns:
        files = list(base_path.glob(pattern))
        if files:
            # Prefer model.pt, then latest epoch, then others
            for f in files:
                if f.name == 'model.pt':
                    return f
            return sorted(files)[-1]
    
    return None


# Valid FFM kernel methods (exclude diffusion methods like DDPM, NCSN, GANO)
VALID_FFM_METHODS = {'none', 'euclidean', 'rbf', 'signature', 'gaussian_ot', 'independent'}


def scan_checkpoint_directory(
    base_dir: Path,
    dataset_name: str = None,
    include_methods: set = None,
) -> Dict[str, Path]:
    """Scan a directory structure for model checkpoints.
    
    Expected structures:
    - base_dir/{method}/seed_{n}/model.pt
    - base_dir/{method}/model.pt
    - base_dir/{method}/{subdir}/seed_{n}/model.pt  (e.g., economy dataset)
    - base_dir/seed_{n}/model.pt
    
    Only scans for valid FFM kernel methods (none, euclidean, rbf, signature).
    Skips diffusion methods (DDPM, NCSN, GANO).
    
    Parameters
    ----------
    base_dir : Path
        Directory to scan
    dataset_name : str, optional
        Dataset name (unused, for future filtering)
    include_methods : set, optional
        Override default methods to include. If None, uses VALID_FFM_METHODS.
    
    Returns
    -------
    Dict mapping method names to checkpoint paths
    """
    base_dir = Path(base_dir)
    checkpoints = {}
    
    if include_methods is None:
        include_methods = VALID_FFM_METHODS
    
    if not base_dir.exists():
        print(f"  ✗ Directory not found: {base_dir}")
        return checkpoints
    
    # Pattern 1: base_dir/{method}/seed_{n}/model.pt or base_dir/{method}/model.pt
    for method_dir in base_dir.iterdir():
        if not method_dir.is_dir():
            continue
        
        method_name = method_dir.name.lower()
        
        # Skip non-method directories
        if method_name.startswith('.') or method_name.startswith('_'):
            continue
        
        # Only include valid FFM kernel methods
        if method_name not in include_methods:
            continue
        
        # Check for direct model file
        model_file = find_model_file(method_dir)
        if model_file:
            checkpoints[method_name] = model_file
            continue
        
        # Check for seed subdirectories directly under method
        found = False
        for seed_dir in method_dir.iterdir():
            if seed_dir.is_dir() and seed_dir.name.startswith('seed_'):
                model_file = find_model_file(seed_dir)
                if model_file:
                    # Use first seed found
                    checkpoints[method_name] = model_file
                    found = True
                    break
        
        if found:
            continue
        
        # Pattern 1b: Check for nested subdirectory structure
        # e.g., base_dir/{method}/{subdir}/seed_{n}/model.pt (economy dataset)
        for subdir in method_dir.iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name.startswith('.') or subdir.name.startswith('_'):
                continue
            if subdir.name.startswith('seed_'):
                # This is actually a seed dir, skip
                continue
            
            # Check for seed dirs inside subdir
            for seed_dir in subdir.iterdir():
                if seed_dir.is_dir() and seed_dir.name.startswith('seed_'):
                    model_file = find_model_file(seed_dir)
                    if model_file:
                        checkpoints[method_name] = model_file
                        found = True
                        break
            
            if found:
                break
    
    # Pattern 2: base_dir/seed_{n}/model.pt (single method)
    if not checkpoints:
        for seed_dir in base_dir.iterdir():
            if seed_dir.is_dir() and seed_dir.name.startswith('seed_'):
                model_file = find_model_file(seed_dir)
                if model_file:
                    checkpoints['model'] = model_file
                    break
    
    return checkpoints


def parse_checkpoint_args(checkpoint_args: List[str]) -> Dict[str, Path]:
    """Parse checkpoint arguments in format 'name=path' or just 'path'.
    
    Examples:
        ['none=./ffm.pt', 'signature=./sig.pt'] -> {'none': Path('./ffm.pt'), ...}
        ['./model.pt'] -> {'model': Path('./model.pt')}
    """
    checkpoints = {}
    
    for arg in checkpoint_args:
        if '=' in arg:
            name, path = arg.split('=', 1)
            checkpoints[name.strip()] = Path(path.strip())
        else:
            # Use filename without extension as name
            path = Path(arg.strip())
            name = path.stem
            # Try to infer method name from parent directories
            for parent in path.parents:
                if parent.name in KERNEL_CONFIGS or parent.name in ['none', 'independent', 'euclidean', 'rbf', 'signature', 'gaussian_ot']:
                    name = parent.name
                    break
            checkpoints[name] = path
    
    return checkpoints


# =============================================================================
# Velocity Field Analysis Functions
# =============================================================================

class VelocityTracker(nn.Module):
    """Wrapper to track velocity norms and NFE during ODE integration."""
    
    def __init__(self, model: nn.Module, record_times: List[float] = None):
        super().__init__()
        self.model = model
        self.record_times = record_times or []
        self.recorded_norms: Dict[float, List[torch.Tensor]] = defaultdict(list)
        self.all_norms: List[Tuple[float, torch.Tensor]] = []
        self.nfe = 0  # Number of function evaluations
        
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with velocity norm recording."""
        self.nfe += 1
        v = self.model(t, x)
        
        # Compute L2 norm over spatial dimensions
        # v shape: (batch, channels, *spatial_dims)
        v_flat = v.view(v.shape[0], -1)  # (batch, n_features)
        norms = torch.norm(v_flat, dim=1)  # (batch,)
        
        t_val = t.item() if t.dim() == 0 else t[0].item()
        self.all_norms.append((t_val, norms.detach().clone()))
        
        # Record at specific times (with tolerance)
        for rec_t in self.record_times:
            if abs(t_val - rec_t) < 0.01:
                self.recorded_norms[rec_t].append(norms.detach().clone())
        
        return v
    
    def reset(self):
        """Reset recorded data."""
        self.recorded_norms = defaultdict(list)
        self.all_norms = []
        self.nfe = 0
    
    def get_all_norms(self) -> List[Tuple[float, torch.Tensor]]:
        """Get all recorded (time, norms) pairs."""
        return self.all_norms
    
    def get_nfe(self) -> int:
        """Get number of function evaluations."""
        return self.nfe


def compute_velocity_norms_along_trajectory(
    model: nn.Module,
    gp: GPPrior,
    dims: Tuple[int, ...],
    n_channels: int = 1,
    n_samples: int = 100,
    n_time_points: int = 50,
    batch_size: int = 50,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    device: str = 'cpu',
) -> TrajectoryAnalysis:
    """
    Compute velocity field norms along ODE trajectories.
    
    Approach 1: Track ||v_θ(t, x_t)|| at multiple time points during integration.
    
    Parameters
    ----------
    model : nn.Module
        Trained velocity field model
    gp : GPPrior
        GP prior for base distribution
    dims : tuple
        Spatial dimensions
    n_channels : int
        Number of channels
    n_samples : int
        Total samples to analyze
    n_time_points : int
        Number of time points to evaluate
    batch_size : int
        Batch size for sampling
    rtol, atol : float
        ODE solver tolerances
    device : str
        
    Returns
    -------
    TrajectoryAnalysis
        Statistics of velocity norms along trajectories
    """
    model = model.to(device)
    model.eval()
    
    # Create evenly spaced time points for evaluation
    t_eval = torch.linspace(0, 1, n_time_points, device=device)
    time_points = t_eval.cpu().numpy().tolist()
    
    # Wrap model with velocity tracker
    tracker = VelocityTracker(model, record_times=time_points)
    
    # Collect norms across all batches
    all_norms_per_time: Dict[int, List[torch.Tensor]] = defaultdict(list)
    nfe_list = []  # Track NFE for each batch
    
    grid = make_grid(dims)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(n_batches):
            current_batch_size = min(batch_size, n_samples - i * batch_size)
            if current_batch_size <= 0:
                break
            
            # Sample from GP prior
            x0 = gp.sample(grid, dims, n_samples=current_batch_size, n_channels=n_channels)
            x0 = x0.to(device)
            
            tracker.reset()
            
            # Integrate ODE - this will record velocity norms at each evaluation
            trajectory = odeint(tracker, x0, t_eval, method='dopri5', rtol=rtol, atol=atol)
            
            # Record NFE for this batch
            nfe_list.append(tracker.get_nfe())
            
            # Collect norms from this batch
            batch_norms = tracker.get_all_norms()
            
            # Organize by closest time point
            for t_val, norms in batch_norms:
                # Find closest time index
                idx = np.argmin(np.abs(np.array(time_points) - t_val))
                all_norms_per_time[idx].append(norms.cpu())
    
    # Compute statistics for each time point
    stats_per_time = []
    for idx in range(len(time_points)):
        if idx in all_norms_per_time and all_norms_per_time[idx]:
            combined_norms = torch.cat(all_norms_per_time[idx])
            stats = VelocityNormStats.from_tensor(combined_norms)
        else:
            # No data for this time point - use zeros
            stats = VelocityNormStats(
                mean=0, std=0, min=0, max=0, median=0,
                q25=0, q75=0, q95=0, q99=0
            )
        stats_per_time.append(stats)
    
    # Compute NFE statistics
    mean_nfe = float(np.mean(nfe_list)) if nfe_list else 0.0
    std_nfe = float(np.std(nfe_list)) if nfe_list else 0.0
    
    analysis = TrajectoryAnalysis(
        time_points=time_points,
        stats_per_time=stats_per_time,
        mean_nfe=mean_nfe,
        std_nfe=std_nfe,
    )
    analysis.compute_aggregates()
    
    return analysis


def compute_velocity_distribution_at_times(
    model: nn.Module,
    gp: GPPrior,
    dims: Tuple[int, ...],
    n_channels: int = 1,
    n_samples: int = 100,
    time_slices: List[float] = None,
    batch_size: int = 50,
    n_bins: int = 50,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    device: str = 'cpu',
) -> DistributionAnalysis:
    """
    Compute velocity norm distributions at specific time slices.
    
    Approach 2: Build histograms of ||v_θ(t, x)|| at fixed t values.
    
    Parameters
    ----------
    model : nn.Module
        Trained velocity field model
    gp : GPPrior
        GP prior for base distribution
    dims : tuple
        Spatial dimensions
    n_channels : int
    n_samples : int
    time_slices : list
        Specific time values to analyze (default: [0.0, 0.25, 0.5, 0.75, 1.0])
    batch_size : int
    n_bins : int
        Number of histogram bins
    rtol, atol : float
        ODE solver tolerances
    device : str
        
    Returns
    -------
    DistributionAnalysis
        Histograms and statistics at each time slice
    """
    if time_slices is None:
        time_slices = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    
    model = model.to(device)
    model.eval()
    
    grid = make_grid(dims)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    # Collect all norms at each time slice
    norms_at_time: Dict[float, List[torch.Tensor]] = defaultdict(list)
    
    with torch.no_grad():
        for i in range(n_batches):
            current_batch_size = min(batch_size, n_samples - i * batch_size)
            if current_batch_size <= 0:
                break
            
            # Sample from GP prior
            x0 = gp.sample(grid, dims, n_samples=current_batch_size, n_channels=n_channels)
            x0 = x0.to(device)
            
            for t_val in time_slices:
                # Integrate to time t
                t_tensor = torch.tensor([0.0, t_val], device=device)
                
                if t_val == 0.0:
                    x_t = x0
                else:
                    trajectory = odeint(model, x0, t_tensor, method='dopri5', 
                                        rtol=rtol, atol=atol)
                    x_t = trajectory[-1]
                
                # Compute velocity at this point
                t_scalar = torch.tensor(t_val, device=device)
                v = model(t_scalar, x_t)
                
                # Compute norms
                v_flat = v.view(v.shape[0], -1)
                norms = torch.norm(v_flat, dim=1)
                norms_at_time[t_val].append(norms.cpu())
    
    # Build histograms and compute statistics
    histograms = {}
    stats_per_time = {}
    
    for t_val in time_slices:
        if t_val in norms_at_time and norms_at_time[t_val]:
            combined_norms = torch.cat(norms_at_time[t_val]).numpy()
            
            # Compute histogram
            counts, bin_edges = np.histogram(combined_norms, bins=n_bins, density=True)
            histograms[t_val] = (counts, bin_edges)
            
            # Compute statistics
            stats_per_time[t_val] = VelocityNormStats.from_tensor(
                torch.tensor(combined_norms)
            )
    
    return DistributionAnalysis(
        time_points=time_slices,
        histograms=histograms,
        stats_per_time=stats_per_time,
    )


# =============================================================================
# Training (for fresh analysis)
# =============================================================================

def load_dataset(dataset_name: str, config: Dict, device: str) -> torch.Tensor:
    """Load dataset for training."""
    data_file = config.get('data_file')
    
    if dataset_name == 'aemet':
        import pandas as pd
        df = pd.read_csv(data_file, index_col=0)
        data = torch.tensor(df.values, dtype=torch.float32).T
        data = data.unsqueeze(1)  # Add channel dim
        
    elif data_file.endswith('.pt'):
        data = torch.load(data_file)
        if data.dim() == 2:
            data = data.unsqueeze(1)
            
    elif data_file.endswith('.mat'):
        import scipy.io
        mat_data = scipy.io.loadmat(data_file)
        # Handle different mat file structures
        for key in ['u', 'data', 'trajectories']:
            if key in mat_data:
                data = torch.tensor(mat_data[key], dtype=torch.float32)
                break
        else:
            # Take first array-like value
            for v in mat_data.values():
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    data = torch.tensor(v, dtype=torch.float32)
                    break
        
        if data.dim() == 2:
            data = data.unsqueeze(1)
    else:
        raise ValueError(f"Unknown data file format: {data_file}")
    
    return data.to(device)


def train_model_quick(
    model: nn.Module,
    data: torch.Tensor,
    ffm_config: Dict,
    dataset_config: Dict,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> nn.Module:
    """Quick training for analysis purposes."""
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create FFM wrapper
    ffm = FFMModelOT(
        model=model,
        kernel_length=dataset_config['kernel_length'],
        kernel_variance=dataset_config['kernel_variance'],
        sigma_min=1e-4,
        device=device,
        dtype=torch.float32,
        **{k: v for k, v in ffm_config.items() if k != 'description'},
    )
    
    # Data loader
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Training loop
    model.train()
    for ep in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            batch_size_actual = batch.shape[0]
            
            # Sample base noise
            z_noise = ffm.sample_base(batch_size_actual, ffm.n_channels if hasattr(ffm, 'n_channels') else 1,
                                      tuple(batch.shape[2:]))
            
            # OT pairing
            x_data, z_paired = ffm.pair_samples(batch, z_noise)
            
            # Sample time
            t = torch.rand(batch_size_actual, device=device)
            
            # Get noisy samples and targets
            x_noisy = ffm.simulate(t, x_data, z_paired)
            target = ffm.get_conditional_fields(t, x_data, x_noisy, z_paired)
            
            # Forward pass
            model_out = model(t, x_noisy)
            
            # Loss and update
            optimizer.zero_grad()
            loss = torch.mean((model_out - target) ** 2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1}/{epochs}: Loss = {epoch_loss/len(loader):.6f}")
    
    model.eval()
    return model


# =============================================================================
# Visualization
# =============================================================================

def plot_trajectory_comparison(
    results: Dict[str, MethodResults],
    output_path: Path,
    title: str = "Velocity Norm Along Trajectories",
):
    """Plot velocity norm statistics over time for multiple methods."""
    # Use a distinctive color palette
    colors = {
        'none': '#2E86AB',      # Blue
        'euclidean': '#A23B72', # Magenta
        'rbf': '#F18F01',       # Orange
        'signature': '#C73E1D', # Red
        'gaussian_ot': '#3B1F2B', # Dark
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean velocity norm over time
    ax1 = axes[0, 0]
    for method_name, result in results.items():
        ta = result.trajectory_analysis
        means = [s.mean for s in ta.stats_per_time]
        stds = [s.std for s in ta.stats_per_time]
        color = colors.get(method_name, '#666666')
        
        ax1.plot(ta.time_points, means, label=KERNEL_CONFIGS.get(method_name, {}).get('description', method_name),
                color=color, linewidth=2)
        ax1.fill_between(ta.time_points, 
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.2, color=color)
    
    ax1.set_xlabel('Flow Time t', fontsize=12)
    ax1.set_ylabel('||v(t, x)||', fontsize=12)
    ax1.set_title('Mean Velocity Norm ± Std', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Max velocity norm over time
    ax2 = axes[0, 1]
    for method_name, result in results.items():
        ta = result.trajectory_analysis
        maxes = [s.max for s in ta.stats_per_time]
        q95s = [s.q95 for s in ta.stats_per_time]
        color = colors.get(method_name, '#666666')
        
        ax2.plot(ta.time_points, maxes, label=f'{KERNEL_CONFIGS.get(method_name, {}).get("description", method_name)} (max)',
                color=color, linewidth=2)
        ax2.plot(ta.time_points, q95s, '--', color=color, linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('Flow Time t', fontsize=12)
    ax2.set_ylabel('||v(t, x)||', fontsize=12)
    ax2.set_title('Max & 95th Percentile Velocity Norm', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Variance of velocity norm over time
    ax3 = axes[1, 0]
    for method_name, result in results.items():
        ta = result.trajectory_analysis
        variances = [s.std ** 2 for s in ta.stats_per_time]
        color = colors.get(method_name, '#666666')
        
        ax3.plot(ta.time_points, variances, 
                label=KERNEL_CONFIGS.get(method_name, {}).get('description', method_name),
                color=color, linewidth=2)
    
    ax3.set_xlabel('Flow Time t', fontsize=12)
    ax3.set_ylabel('Var(||v(t, x)||)', fontsize=12)
    ax3.set_title('Velocity Norm Variance Over Time', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary bar chart
    ax4 = axes[1, 1]
    methods = list(results.keys())
    x = np.arange(len(methods))
    width = 0.25
    
    overall_means = [results[m].trajectory_analysis.overall_mean for m in methods]
    overall_maxes = [results[m].trajectory_analysis.overall_max for m in methods]
    time_vars = [results[m].trajectory_analysis.time_variance for m in methods]
    
    # Normalize time variance for visualization
    max_tv = max(time_vars) if max(time_vars) > 0 else 1
    time_vars_norm = [tv / max_tv * max(overall_means) for tv in time_vars]
    
    bars1 = ax4.bar(x - width, overall_means, width, label='Overall Mean', color='#2E86AB')
    bars2 = ax4.bar(x, overall_maxes, width, label='Overall Max', color='#C73E1D')
    bars3 = ax4.bar(x + width, time_vars_norm, width, label='Time Variance (scaled)', color='#F18F01', alpha=0.7)
    
    ax4.set_xlabel('Method', fontsize=12)
    ax4.set_ylabel('Velocity Norm', fontsize=12)
    ax4.set_title('Summary Statistics', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([KERNEL_CONFIGS.get(m, {}).get('description', m)[:12] for m in methods], 
                        rotation=15, ha='right', fontsize=9)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved trajectory plot: {output_path}")


def plot_distribution_comparison(
    results: Dict[str, MethodResults],
    output_path: Path,
    title: str = "Velocity Norm Distributions",
):
    """Plot velocity norm distributions at different time slices."""
    colors = {
        'none': '#2E86AB',
        'euclidean': '#A23B72',
        'rbf': '#F18F01',
        'signature': '#C73E1D',
        'gaussian_ot': '#3B1F2B',
    }
    
    # Get time slices from first result
    first_result = next(iter(results.values()))
    time_slices = first_result.distribution_analysis.time_points
    
    # Create subplot for each time slice
    n_times = len(time_slices)
    n_cols = min(4, n_times)
    n_rows = (n_times + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, t_val in enumerate(time_slices):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        for method_name, result in results.items():
            da = result.distribution_analysis
            if t_val in da.histograms:
                counts, bin_edges = da.histograms[t_val]
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                color = colors.get(method_name, '#666666')
                
                ax.plot(bin_centers, counts, 
                       label=KERNEL_CONFIGS.get(method_name, {}).get('description', method_name),
                       color=color, linewidth=2)
                ax.fill_between(bin_centers, counts, alpha=0.15, color=color)
        
        ax.set_xlabel('||v||', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f't = {t_val:.2f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='best', fontsize=8)
    
    # Hide empty subplots
    for idx in range(n_times, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved distribution plot: {output_path}")


def plot_stability_summary(
    results: Dict[str, MethodResults],
    output_path: Path,
    title: str = "Sampling Stability Summary",
):
    """Create a summary plot focused on stability indicators."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    methods = list(results.keys())
    method_labels = [KERNEL_CONFIGS.get(m, {}).get('description', m) for m in methods]
    x = np.arange(len(methods))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6B4C9A', '#1E8449']
    method_colors = [colors[i % len(colors)] for i in range(len(methods))]
    
    # Plot 1: Max velocity norm (stability indicator)
    ax1 = axes[0, 0]
    max_norms = [results[m].trajectory_analysis.overall_max for m in methods]
    bars = ax1.bar(x, max_norms, color=method_colors, edgecolor='black', linewidth=1.2)
    ax1.set_xlabel('Method', fontsize=11)
    ax1.set_ylabel('Max ||v||', fontsize=11)
    ax1.set_title('Maximum Velocity Norm\n(Lower = More Stable)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=20, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, max_norms):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(max_norms),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Time variance of mean velocity (flow uniformity)
    ax2 = axes[0, 1]
    time_vars = [results[m].trajectory_analysis.time_variance for m in methods]
    bars = ax2.bar(x, time_vars, color=method_colors, edgecolor='black', linewidth=1.2)
    ax2.set_xlabel('Method', fontsize=11)
    ax2.set_ylabel('Var(mean ||v|| over t)', fontsize=11)
    ax2.set_title('Velocity Uniformity Over Time\n(Lower = Straighter Paths)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_labels, rotation=20, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    if max(time_vars) > 0:
        for bar, val in zip(bars, time_vars):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(time_vars),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: NFE (Number of Function Evaluations)
    ax3 = axes[1, 0]
    nfes = [results[m].trajectory_analysis.mean_nfe for m in methods]
    nfe_stds = [results[m].trajectory_analysis.std_nfe for m in methods]
    bars = ax3.bar(x, nfes, yerr=nfe_stds, color=method_colors, edgecolor='black', linewidth=1.2,
                   capsize=3, error_kw={'linewidth': 1.5})
    ax3.set_xlabel('Method', fontsize=11)
    ax3.set_ylabel('NFE (Integration Steps)', fontsize=11)
    ax3.set_title('Number of Function Evaluations\n(Lower = Faster Sampling)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(method_labels, rotation=20, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    if max(nfes) > 0:
        for bar, val in zip(bars, nfes):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02*max(nfes),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: 99th percentile at t=0.5 (typical case stability)
    ax4 = axes[1, 1]
    q99_mid = []
    for m in methods:
        da = results[m].distribution_analysis
        if da.stats_per_time:
            # Find closest time to 0.5
            closest_t = min(da.stats_per_time.keys(), key=lambda t: abs(t - 0.5))
            q99_mid.append(da.stats_per_time[closest_t].q99)
        else:
            q99_mid.append(0)
    
    bars = ax4.bar(x, q99_mid, color=method_colors, edgecolor='black', linewidth=1.2)
    ax4.set_xlabel('Method', fontsize=11)
    ax4.set_ylabel('99th Percentile ||v||', fontsize=11)
    ax4.set_title('99th Percentile at t=0.5\n(Tail Behavior)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(method_labels, rotation=20, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    if max(q99_mid) > 0:
        for bar, val in zip(bars, q99_mid):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(q99_mid),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved stability summary: {output_path}")


def plot_stability_summary_aggregated(
    aggregated: Dict[str, AggregatedResults],
    output_path: Path,
    title: str = "Sampling Stability Summary",
):
    """Create a summary plot with aggregated results (mean ± std) from multiple runs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = list(aggregated.keys())
    method_labels = [KERNEL_CONFIGS.get(m, {}).get('description', m) for m in methods]
    x = np.arange(len(methods))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#6B4C9A', '#1E8449']
    method_colors = [colors[i % len(colors)] for i in range(len(methods))]
    
    # Plot 1: Flow Uniformity (time variance) with error bars
    ax1 = axes[0]
    uniformities = [aggregated[m].uniformity_mean for m in methods]
    uniformity_stds = [aggregated[m].uniformity_std for m in methods]
    
    bars = ax1.bar(x, uniformities, yerr=uniformity_stds, color=method_colors, 
                   edgecolor='black', linewidth=1.2, capsize=5, error_kw={'linewidth': 2})
    ax1.set_xlabel('Method', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Flow Uniformity\n(Time Variance)', fontsize=12, fontweight='bold')
    ax1.set_title('Flow Uniformity\n(Lower = Straighter Paths)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_labels, rotation=20, ha='right', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))
    
    # Add value labels
    if max(uniformities) > 0:
        for bar, val, std in zip(bars, uniformities, uniformity_stds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02*max(uniformities),
                    f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: NFE with error bars
    ax2 = axes[1]
    nfes = [aggregated[m].nfe_mean for m in methods]
    nfe_stds = [aggregated[m].nfe_std for m in methods]
    
    bars = ax2.bar(x, nfes, yerr=nfe_stds, color=method_colors, edgecolor='black', 
                   linewidth=1.2, capsize=5, error_kw={'linewidth': 2})
    ax2.set_xlabel('Method', fontsize=13, fontweight='bold')
    ax2.set_ylabel('NFE (Integration Steps)', fontsize=12, fontweight='bold')
    ax2.set_title('Number of Function Evaluations\n(Lower = Faster Sampling)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_labels, rotation=20, ha='right', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    if max(nfes) > 0:
        for bar, val, std in zip(bars, nfes, nfe_stds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02*max(nfes),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    
    # Add n_runs info
    n_runs = aggregated[methods[0]].n_runs if methods else 0
    fig.text(0.5, 0.01, f'Results averaged over {n_runs} runs (error bars show ±1 std)', 
             ha='center', fontsize=10, style='italic')
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved aggregated stability summary: {output_path}")


# =============================================================================
# Multi-run Analysis Functions
# =============================================================================

def run_single_analysis(
    model: nn.Module,
    gp: GPPrior,
    dims: Tuple[int, ...],
    n_channels: int,
    n_samples: int,
    n_time_points: int,
    batch_size: int,
    device: str,
    seed: int = None,
) -> TrajectoryAnalysis:
    """Run a single trajectory analysis with optional seed."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    return compute_velocity_norms_along_trajectory(
        model=model,
        gp=gp,
        dims=dims,
        n_channels=n_channels,
        n_samples=n_samples,
        n_time_points=n_time_points,
        batch_size=batch_size,
        device=device,
    )


def run_multiple_analyses(
    model: nn.Module,
    gp: GPPrior,
    dims: Tuple[int, ...],
    n_channels: int,
    n_samples: int,
    n_time_points: int,
    batch_size: int,
    device: str,
    n_runs: int = 10,
    base_seed: int = 42,
    verbose: bool = True,
) -> List[TrajectoryAnalysis]:
    """Run trajectory analysis multiple times with different seeds."""
    analyses = []
    
    for run_idx in range(n_runs):
        seed = base_seed + run_idx * 1000
        if verbose:
            print(f"      Run {run_idx + 1}/{n_runs} (seed={seed})...", end=" ")
        
        analysis = run_single_analysis(
            model=model,
            gp=gp,
            dims=dims,
            n_channels=n_channels,
            n_samples=n_samples,
            n_time_points=n_time_points,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        analyses.append(analysis)
        
        if verbose:
            print(f"NFE={analysis.mean_nfe:.0f}, Uniformity={analysis.time_variance:.6f}")
    
    return analyses


# =============================================================================
# Checkpoint-based Analysis (Recommended)
# =============================================================================

def analyze_from_checkpoints(
    checkpoints: Dict[str, Path],
    dataset_name: str,
    n_samples: int = 100,
    n_time_points: int = 50,
    time_slices: List[float] = None,
    batch_size: int = 64,
    n_runs: int = 10,
    output_dir: Path = None,
    device: str = 'cpu',
) -> Tuple[Dict[str, MethodResults], Dict[str, AggregatedResults]]:
    """
    Analyze velocity fields from pre-trained model checkpoints.
    
    Runs analysis multiple times and reports mean ± std for NFE and flow uniformity.
    
    Parameters
    ----------
    checkpoints : dict
        Mapping from method name to checkpoint path
        e.g., {'none': Path('ffm.pt'), 'signature': Path('sig.pt')}
    dataset_name : str
        Dataset name (for GP prior and model config)
    n_samples : int
        Number of samples for analysis per run
    n_time_points : int
        Number of time points for trajectory analysis
    time_slices : list
        Time values for distribution analysis
    batch_size : int
    n_runs : int
        Number of runs for computing mean ± std (default: 10)
    output_dir : Path
    device : str
        
    Returns
    -------
    results : dict
        MethodResults for each loaded model (from last run)
    aggregated : dict
        AggregatedResults with mean ± std across runs
    """
    if time_slices is None:
        time_slices = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    
    if output_dir is None:
        output_dir = Path(f'../outputs/velocity_analysis/{dataset_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataset config
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    ds_config = DATASET_CONFIGS[dataset_name]
    dims = ds_config['spatial_dims']
    n_channels = ds_config['n_channels']
    
    print(f"="*70)
    print(f"Velocity Field Analysis from Checkpoints: {dataset_name}")
    print(f"="*70)
    print(f"  Spatial dims: {dims}")
    print(f"  Checkpoints to analyze: {list(checkpoints.keys())}")
    print(f"  Samples per run: {n_samples}, Time points: {n_time_points}")
    print(f"  Number of runs: {n_runs}")
    print(f"  Device: {device}")
    print()
    
    # Create GP prior
    gp = GPPrior(
        lengthscale=ds_config['kernel_length'],
        var=ds_config['kernel_variance'],
        device=device,
    )
    
    results = {}
    aggregated = {}
    
    for method_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*50}")
        print(f"Analyzing: {method_name}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*50}")
        
        # Load model from checkpoint
        model = load_model_checkpoint(ckpt_path, ds_config, device)
        
        if model is None:
            print(f"  ⚠ Skipping {method_name} - could not load checkpoint")
            continue
        
        model.eval()
        
        # Run trajectory analysis multiple times
        print(f"  Running {n_runs} trajectory analyses...")
        trajectory_analyses = run_multiple_analyses(
            model=model,
            gp=gp,
            dims=dims,
            n_channels=n_channels,
            n_samples=n_samples,
            n_time_points=n_time_points,
            batch_size=min(batch_size, n_samples),
            device=device,
            n_runs=n_runs,
            verbose=True,
        )
        
        # Aggregate results
        agg_result = AggregatedResults.from_trajectory_analyses(method_name, trajectory_analyses)
        aggregated[method_name] = agg_result
        
        print(f"\n    === Aggregated Results ({n_runs} runs) ===")
        print(f"    Flow Uniformity: {agg_result.uniformity_mean:.6f} ± {agg_result.uniformity_std:.6f}")
        print(f"    NFE: {agg_result.nfe_mean:.1f} ± {agg_result.nfe_std:.1f}")
        
        # Run distribution analysis (single run for visualization)
        print(f"  Running distribution analysis (single run for plots)...")
        distribution_analysis = compute_velocity_distribution_at_times(
            model=model,
            gp=gp,
            dims=dims,
            n_channels=n_channels,
            n_samples=n_samples,
            time_slices=time_slices,
            batch_size=min(batch_size, n_samples),
            device=device,
        )
        
        # Get kernel config if known
        kernel_config = KERNEL_CONFIGS.get(method_name, {'description': method_name})
        
        # Store results (use last trajectory analysis for detailed stats)
        results[method_name] = MethodResults(
            method_name=method_name,
            trajectory_analysis=trajectory_analyses[-1],
            distribution_analysis=distribution_analysis,
            config=kernel_config,
        )
    
    if not results:
        print("\n⚠ No models were successfully loaded!")
        return results, aggregated
    
    # Generate plots
    print(f"\n{'='*50}")
    print("Generating comparison plots...")
    print(f"{'='*50}")
    
    plot_trajectory_comparison(
        results,
        output_dir / f'{dataset_name}_velocity_trajectory.pdf',
        title=f'{dataset_name.upper()}: Velocity Norms Along Trajectories',
    )
    
    plot_distribution_comparison(
        results,
        output_dir / f'{dataset_name}_velocity_distribution.pdf',
        title=f'{dataset_name.upper()}: Velocity Norm Distributions',
    )
    
    # Use aggregated results for stability summary
    plot_stability_summary_aggregated(
        aggregated,
        output_dir / f'{dataset_name}_stability_summary.pdf',
        title=f'{dataset_name.upper()}: Sampling Stability ({n_runs} runs)',
    )
    
    # Save numerical results
    results_dict = {}
    for method_name, result in results.items():
        ta = result.trajectory_analysis
        da = result.distribution_analysis
        agg = aggregated.get(method_name)
        
        results_dict[method_name] = {
            'checkpoint': str(checkpoints.get(method_name, 'unknown')),
            'config': result.config,
            'aggregated': {
                'n_runs': agg.n_runs if agg else 1,
                'uniformity_mean': agg.uniformity_mean if agg else ta.time_variance,
                'uniformity_std': agg.uniformity_std if agg else 0.0,
                'nfe_mean': agg.nfe_mean if agg else ta.mean_nfe,
                'nfe_std': agg.nfe_std if agg else ta.std_nfe,
                'uniformity_values': agg.uniformity_values if agg else [ta.time_variance],
                'nfe_values': agg.nfe_values if agg else [ta.mean_nfe],
            },
            'trajectory': {
                'time_variance': ta.time_variance,
                'mean_nfe': ta.mean_nfe,
                'time_points': ta.time_points,
            },
            'distribution': {
                'time_points': da.time_points,
                'stats_per_time': {str(k): asdict(v) for k, v in da.stats_per_time.items()},
            },
        }
    
    with open(output_dir / f'{dataset_name}_velocity_analysis.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    return results, aggregated


# =============================================================================
# Main Analysis Pipeline (Training-based)
# =============================================================================

def analyze_velocity_field(
    dataset_name: str,
    kernels: List[str],
    n_samples: int = 100,
    n_time_points: int = 50,
    time_slices: List[float] = None,
    epochs: int = 50,
    batch_size: int = 64,
    train_models: bool = True,
    model_dir: Optional[Path] = None,
    output_dir: Path = None,
    device: str = 'cpu',
) -> Dict[str, MethodResults]:
    """
    Run complete velocity field analysis.
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset to analyze
    kernels : list
        List of kernel configurations to compare
    n_samples : int
        Number of samples for analysis
    n_time_points : int
        Number of time points for trajectory analysis
    time_slices : list
        Time values for distribution analysis
    epochs : int
        Training epochs (if training models)
    batch_size : int
    train_models : bool
        Whether to train models or load existing
    model_dir : Path
        Directory to load models from (if not training)
    output_dir : Path
        Directory to save results
    device : str
        
    Returns
    -------
    results : dict
        MethodResults for each kernel configuration
    """
    if time_slices is None:
        time_slices = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    
    if output_dir is None:
        output_dir = Path(f'../outputs/velocity_analysis/{dataset_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get dataset config
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    ds_config = DATASET_CONFIGS[dataset_name]
    dims = ds_config['spatial_dims']
    n_channels = ds_config['n_channels']
    
    print(f"="*70)
    print(f"Velocity Field Analysis: {dataset_name}")
    print(f"="*70)
    print(f"  Spatial dims: {dims}")
    print(f"  Kernels to analyze: {kernels}")
    print(f"  Samples: {n_samples}, Time points: {n_time_points}")
    print(f"  Device: {device}")
    print()
    
    # Create GP prior
    gp = GPPrior(
        lengthscale=ds_config['kernel_length'],
        var=ds_config['kernel_variance'],
        device=device,
    )
    
    # Load dataset if training
    if train_models:
        try:
            data = load_dataset(dataset_name, ds_config, device)
            print(f"  Loaded dataset: {data.shape}")
        except Exception as e:
            print(f"  Warning: Could not load dataset ({e}), using synthetic data")
            data = gp.sample(make_grid(dims), dims, n_samples=500, n_channels=n_channels)
            data = data.to(device)
    
    results = {}
    
    for kernel_name in kernels:
        print(f"\n{'='*50}")
        print(f"Analyzing kernel: {kernel_name}")
        print(f"{'='*50}")
        
        if kernel_name not in KERNEL_CONFIGS:
            print(f"  Warning: Unknown kernel config '{kernel_name}', skipping")
            continue
        
        kernel_config = KERNEL_CONFIGS[kernel_name]
        
        # Create or load model
        model = create_model(ds_config, device)
        
        if train_models:
            print(f"  Training model ({epochs} epochs)...")
            model = train_model_quick(
                model=model,
                data=data,
                ffm_config=kernel_config,
                dataset_config=ds_config,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
            )
        elif model_dir is not None:
            # Try to load from directory
            method_subdir = model_dir / kernel_name
            if method_subdir.exists():
                model_path = find_model_file(method_subdir)
                if model_path:
                    loaded_model = load_model_checkpoint(model_path, ds_config, device)
                    if loaded_model is not None:
                        model = loaded_model
                        print(f"  Loaded model from {model_path}")
                    else:
                        print(f"  Could not load model, using random init")
                else:
                    print(f"  No model file found, using random init")
            else:
                print(f"  Model directory not found, using random init")
        else:
            print(f"  Using randomly initialized model")
        
        model.eval()
        
        # Run trajectory analysis (Approach 1)
        print(f"  Running trajectory analysis...")
        trajectory_analysis = compute_velocity_norms_along_trajectory(
            model=model,
            gp=gp,
            dims=dims,
            n_channels=n_channels,
            n_samples=n_samples,
            n_time_points=n_time_points,
            batch_size=min(batch_size, n_samples),
            device=device,
        )
        
        print(f"    Overall max: {trajectory_analysis.overall_max:.4f}")
        print(f"    Overall mean: {trajectory_analysis.overall_mean:.4f}")
        print(f"    Time variance: {trajectory_analysis.time_variance:.6f}")
        print(f"    NFE (integration steps): {trajectory_analysis.mean_nfe:.1f} ± {trajectory_analysis.std_nfe:.1f}")
        
        # Run distribution analysis (Approach 2)
        print(f"  Running distribution analysis...")
        distribution_analysis = compute_velocity_distribution_at_times(
            model=model,
            gp=gp,
            dims=dims,
            n_channels=n_channels,
            n_samples=n_samples,
            time_slices=time_slices,
            batch_size=min(batch_size, n_samples),
            device=device,
        )
        
        for t_val, stats in distribution_analysis.stats_per_time.items():
            print(f"    t={t_val:.2f}: mean={stats.mean:.4f}, max={stats.max:.4f}, q95={stats.q95:.4f}")
        
        # Store results
        results[kernel_name] = MethodResults(
            method_name=kernel_name,
            trajectory_analysis=trajectory_analysis,
            distribution_analysis=distribution_analysis,
            config=kernel_config,
        )
    
    # Generate plots
    print(f"\n{'='*50}")
    print("Generating comparison plots...")
    print(f"{'='*50}")
    
    plot_trajectory_comparison(
        results,
        output_dir / f'{dataset_name}_velocity_trajectory.pdf',
        title=f'{dataset_name.upper()}: Velocity Norms Along Trajectories',
    )
    
    plot_distribution_comparison(
        results,
        output_dir / f'{dataset_name}_velocity_distribution.pdf',
        title=f'{dataset_name.upper()}: Velocity Norm Distributions',
    )
    
    plot_stability_summary(
        results,
        output_dir / f'{dataset_name}_stability_summary.pdf',
        title=f'{dataset_name.upper()}: Sampling Stability Summary',
    )
    
    # Save numerical results
    results_dict = {}
    for method_name, result in results.items():
        ta = result.trajectory_analysis
        da = result.distribution_analysis
        
        results_dict[method_name] = {
            'config': result.config,
            'trajectory': {
                'overall_max': ta.overall_max,
                'overall_mean': ta.overall_mean,
                'time_variance': ta.time_variance,
                'mean_nfe': ta.mean_nfe,
                'std_nfe': ta.std_nfe,
                'time_points': ta.time_points,
                'stats_per_time': [asdict(s) for s in ta.stats_per_time],
            },
            'distribution': {
                'time_points': da.time_points,
                'stats_per_time': {str(k): asdict(v) for k, v in da.stats_per_time.items()},
            },
        }
    
    with open(output_dir / f'{dataset_name}_velocity_analysis.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze velocity field norm distributions for flow matching models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # ===== RECOMMENDED: Analyze pre-trained checkpoints =====
    
    # Single checkpoint
    python analyze_velocity_field.py --checkpoint path/to/model.pt --dataset aemet
    
    # Multiple checkpoints (compare methods)
    python analyze_velocity_field.py --checkpoint none=./ffm.pt signature=./sig.pt rbf=./rbf.pt --dataset aemet
    
    # Scan directory for all method checkpoints
    python analyze_velocity_field.py --scan-dir ../outputs/seeded_runs/aemet --dataset aemet
    
    # ===== Alternative: Train fresh models =====
    
    # Quick test with random models
    python analyze_velocity_field.py --no-train --n-samples 50
    
    # Full analysis for AEMET
    python analyze_velocity_field.py --dataset aemet --epochs 100
    
    # Compare specific kernels
    python analyze_velocity_field.py --kernels none signature rbf --epochs 50
        """,
    )
    
    # Checkpoint loading arguments (recommended)
    ckpt_group = parser.add_argument_group('Checkpoint Loading (Recommended)')
    ckpt_group.add_argument('--checkpoint', type=str, nargs='+', default=None,
                           help='Checkpoint path(s). Format: "path" or "name=path". '
                                'Examples: model.pt, none=ffm.pt signature=sig.pt')
    ckpt_group.add_argument('--scan-dir', type=str, default=None,
                           help='Scan directory for model checkpoints. '
                                'Expects structure: dir/{method}/seed_*/model.pt')
    
    # Dataset and model config
    config_group = parser.add_argument_group('Dataset Configuration')
    config_group.add_argument('--dataset', type=str, nargs='+', default=['aemet'],
                             choices=list(DATASET_CONFIGS.keys()),
                             help='Dataset(s) to analyze (for GP prior and model config)')
    
    # Training arguments (alternative to checkpoints)
    train_group = parser.add_argument_group('Training (Alternative)')
    train_group.add_argument('--kernels', type=str, nargs='+', 
                            default=['none', 'euclidean', 'rbf', 'signature'],
                            choices=list(KERNEL_CONFIGS.keys()),
                            help='Kernel configurations to train and compare')
    train_group.add_argument('--epochs', type=int, default=50,
                            help='Training epochs')
    train_group.add_argument('--no-train', action='store_true',
                            help='Skip training, use random models (for testing)')
    train_group.add_argument('--load-models', action='store_true',
                            help='[DEPRECATED] Use --checkpoint or --scan-dir instead')
    train_group.add_argument('--model-dir', type=str, default=None,
                            help='[DEPRECATED] Use --scan-dir instead')
    
    # Analysis parameters
    analysis_group = parser.add_argument_group('Analysis Parameters')
    analysis_group.add_argument('--n-samples', type=int, default=100,
                               help='Number of samples for analysis per run')
    analysis_group.add_argument('--n-time-points', type=int, default=50,
                               help='Number of time points for trajectory analysis')
    analysis_group.add_argument('--batch-size', type=int, default=64,
                               help='Batch size')
    analysis_group.add_argument('--n-runs', type=int, default=10,
                               help='Number of runs for computing mean ± std')
    
    # Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', type=str, default='../outputs/velocity_analysis',
                             help='Output directory for results')
    output_group.add_argument('--device', type=str, default=None,
                             help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*70)
    print("Velocity Field Norm Analysis")
    print("="*70)
    print(f"Device: {device}")
    
    # Determine mode: checkpoint loading vs training
    use_checkpoints = args.checkpoint is not None or args.scan_dir is not None
    
    if use_checkpoints:
        # ===== CHECKPOINT MODE =====
        print("Mode: Analyzing pre-trained checkpoints")
        
        # Only use first dataset for checkpoint mode
        dataset_name = args.dataset[0]
        
        # Collect checkpoints
        checkpoints = {}
        
        if args.checkpoint:
            checkpoints = parse_checkpoint_args(args.checkpoint)
            print(f"Checkpoints from arguments: {list(checkpoints.keys())}")
        
        if args.scan_dir:
            scan_path = Path(args.scan_dir)
            scanned = scan_checkpoint_directory(scan_path, dataset_name)
            print(f"Checkpoints from scan: {list(scanned.keys())}")
            
            # Report skipped directories
            if scan_path.exists():
                all_dirs = {d.name.lower() for d in scan_path.iterdir() if d.is_dir() and not d.name.startswith('.')}
                skipped = all_dirs - set(scanned.keys()) - VALID_FFM_METHODS
                if skipped:
                    print(f"  (Skipped non-FFM methods: {', '.join(sorted(skipped))})")
            
            checkpoints.update(scanned)
        
        if not checkpoints:
            print("\n✗ No checkpoints found! Please provide --checkpoint or valid --scan-dir")
            return
        
        print(f"\nCheckpoints to analyze:")
        for name, path in checkpoints.items():
            exists = "✓" if path.exists() else "✗"
            print(f"  {exists} {name}: {path}")
        print()
        
        output_dir = Path(args.output_dir) / dataset_name
        
        results, aggregated = analyze_from_checkpoints(
            checkpoints=checkpoints,
            dataset_name=dataset_name,
            n_samples=args.n_samples,
            n_time_points=args.n_time_points,
            batch_size=args.batch_size,
            n_runs=args.n_runs,
            output_dir=output_dir,
            device=device,
        )
        
    else:
        # ===== TRAINING MODE =====
        print("Mode: Training fresh models")
        print(f"Datasets: {args.dataset}")
        print(f"Kernels: {args.kernels}")
        print(f"Training: {'No (random models)' if args.no_train else f'Yes ({args.epochs} epochs)'}")
        print()
        
        # Handle deprecated arguments
        if args.load_models or args.model_dir:
            print("⚠ Warning: --load-models and --model-dir are deprecated.")
            print("  Use --checkpoint or --scan-dir instead for loading trained models.")
            print()
        
        aggregated = {}  # Initialize for summary table
        
        # Run analysis for each dataset
        for dataset_name in args.dataset:
            output_dir = Path(args.output_dir) / dataset_name
            model_dir = Path(args.model_dir) if args.model_dir else None
            
            results = analyze_velocity_field(
                dataset_name=dataset_name,
                kernels=args.kernels,
                n_samples=args.n_samples,
                n_time_points=args.n_time_points,
                epochs=args.epochs,
                batch_size=args.batch_size,
                train_models=not args.no_train and not args.load_models,
                model_dir=model_dir,
                output_dir=output_dir,
                device=device,
            )
    
    # Print summary table
    if use_checkpoints and aggregated:
        # Use aggregated results for checkpoint mode
        print(f"\n{'='*85}")
        print(f"SUMMARY ({args.n_runs} runs)")
        print(f"{'='*85}")
        print(f"{'Method':<15} {'Flow Uniformity':>25} {'NFE':>25} {'Rank':>10}")
        print(f"{'':<15} {'(lower = straighter)':>25} {'(lower = faster)':>25} {'':<10}")
        print("-"*85)
        
        # Compute ranking based on uniformity (lower is better)
        methods_sorted = sorted(aggregated.keys(), 
                               key=lambda m: aggregated[m].uniformity_mean)
        rankings = {m: i+1 for i, m in enumerate(methods_sorted)}
        
        for method_name, agg in aggregated.items():
            uniformity_str = f"{agg.uniformity_mean:.2e} ± {agg.uniformity_std:.2e}"
            nfe_str = f"{agg.nfe_mean:.1f} ± {agg.nfe_std:.1f}"
            rank = rankings[method_name]
            rank_str = "★" * (len(aggregated) - rank + 1)
            print(f"{method_name:<15} {uniformity_str:>25} {nfe_str:>25} {rank_str:>10}")
        
        print()
        print("Flow Uniformity = Variance of mean velocity norm over time")
        print("NFE = Number of Function Evaluations (integration steps)")
        print("Rank: More stars = lower uniformity = straighter paths")
        print()
        
    elif results:
        # Use single-run results for training mode
        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"{'Method':<15} {'Flow Uniformity':>20} {'NFE':>15}")
        print("-"*70)
        
        for method_name, result in results.items():
            ta = result.trajectory_analysis
            uniformity_str = f"{ta.time_variance:.6f}"
            nfe_str = f"{ta.mean_nfe:.0f}" if ta.mean_nfe > 0 else "-"
            print(f"{method_name:<15} {uniformity_str:>20} {nfe_str:>15}")
        
        print()
        print("Flow Uniformity = Variance of mean velocity norm over time (lower = straighter)")
        print("NFE = Number of Function Evaluations (lower = faster)")
        print()


if __name__ == '__main__':
    main()
