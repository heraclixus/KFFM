#!/usr/bin/env python3
"""
Generate samples from existing trained models.

This script scans output directories for trained models (model.pt) where
samples.pt doesn't exist (or is missing), loads the model, and generates samples.

Supports:
- FFM/k-FFM models (various kernel configurations)
- DDPM (Denoising Diffusion Probabilistic Model)
- NCSN (Noise Conditional Score Network / DDO)
- GANO (Generative Adversarial Neural Operator)

Usage:
    # Scan all datasets and generate missing samples
    python generate_samples.py
    
    # Generate for specific dataset
    python generate_samples.py --dataset navier_stokes
    
    # Force regenerate even if samples exist
    python generate_samples.py --dataset AEMET --force
    
    # Generate specific number of samples
    python generate_samples.py --n-samples 500
    
    # List what would be generated (dry run)
    python generate_samples.py --dry-run
"""

import sys
sys.path.append('../')

import argparse
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

from models.fno import FNO
from functional_fm_ot import FFMModelOT
from diffusion import DiffusionModel

# =============================================================================
# Dataset configurations
# =============================================================================

DATASETS = {
    # 1D Sequence datasets
    'AEMET': {
        'dir': 'AEMET_ot_comprehensive',
        'is_2d': False,
        'spatial_dims': (365,),
        'n_channels': 1,
        'modes': 64,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        't_scaling': 1000,
    },
    'expr_genes': {
        'dir': 'expr_genes_ot_comprehensive',
        'is_2d': False,
        'spatial_dims': (64,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        't_scaling': 1000,
    },
    'economy': {
        'dir': 'econ_ot_comprehensive',
        'is_2d': False,
        'spatial_dims': (128,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        't_scaling': 1000,
        'subdirs': ['econ1_population', 'econ2_gdp', 'econ3_labor'],
    },
    'Heston': {
        'dir': 'Heston_ot_kappa1.0',
        'is_2d': False,
        'spatial_dims': (64,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        't_scaling': 1000,
    },
    'Heston_long': {
        'dir': 'Heston_ot_long',
        'is_2d': False,
        'spatial_dims': (1000,),
        'n_channels': 1,
        'modes': 64,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        't_scaling': 1000,
    },
    'rBergomi': {
        'dir': 'rBergomi_ot_H0p10',
        'is_2d': False,
        'spatial_dims': (64,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        't_scaling': 1000,
    },
    'rBergomi_long': {
        'dir': 'rBergomi_ot_long',
        'is_2d': False,
        'spatial_dims': (1000,),
        'n_channels': 1,
        'modes': 64,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.001,
        'kernel_variance': 1.0,
        't_scaling': 1000,
    },
    # 1D PDE datasets
    'kdv': {
        'dir': 'kdv_ot',
        'is_2d': False,
        'spatial_dims': (512,),
        'n_channels': 1,
        'modes': 64,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        't_scaling': 1000,
    },
    'stochastic_kdv': {
        'dir': 'stochastic_kdv_ot',
        'is_2d': False,
        'spatial_dims': (128,),
        'n_channels': 1,
        'modes': 32,
        'width': 256,
        'mlp_width': 128,
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        't_scaling': 1000,
    },
    # 2D PDE datasets
    'navier_stokes': {
        'dir': 'navier_stokes_ot',
        'is_2d': True,
        'spatial_dims': (64, 64),
        'n_channels': 1,
        'modes': 16,
        'hch': 64,  # hidden channels
        'pch': 128,  # proj channels
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        't_scaling': 1000,
    },
    'stochastic_ns': {
        'dir': 'stochastic_ns_ot',
        'is_2d': True,
        'spatial_dims': (64, 64),
        'n_channels': 1,
        'modes': 16,
        'hch': 64,
        'pch': 128,
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        't_scaling': 1000,
    },
    'ginzburg_landau': {
        'dir': 'ginzburg_landau_ot',
        'is_2d': True,
        'spatial_dims': (64, 64),
        'n_channels': 1,
        'modes': 16,
        'hch': 64,
        'pch': 128,
        'kernel_length': 0.01,
        'kernel_variance': 0.1,
        't_scaling': 1000,
    },
}

# Method type detection from config name
def detect_method_type(config_name: str) -> str:
    """Detect the method type from config name."""
    config_lower = config_name.lower()
    if 'ddpm' in config_lower:
        return 'DDPM'
    elif 'ncsn' in config_lower or 'ddo' in config_lower:
        return 'NCSN'
    elif 'gano' in config_lower:
        return 'GANO'
    else:
        return 'FFM'  # Default to FFM for all k-FFM variants


# =============================================================================
# Model loading functions
# =============================================================================

def create_model_1d(modes: int, width: int, mlp_width: int, t_scaling: float, device: str) -> FNO:
    """Create 1D FNO model."""
    return FNO(
        modes,
        vis_channels=1,
        hidden_channels=width,
        proj_channels=mlp_width,
        x_dim=1,
        t_scaling=t_scaling,
    ).to(device)


def create_model_2d(modes: int, hch: int, pch: int, t_scaling: float, device: str) -> FNO:
    """Create 2D FNO model."""
    return FNO(
        modes,
        vis_channels=1,
        hidden_channels=hch,
        proj_channels=pch,
        x_dim=2,
        t_scaling=t_scaling,
    ).to(device)


def load_state_dict_flexible(model: FNO, checkpoint_path: Path, device: str) -> bool:
    """Load state dict with flexible format handling."""
    try:
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(loaded, dict):
            # Check for wrapped formats
            if 'model' in loaded and isinstance(loaded['model'], dict):
                state_dict = loaded['model']
            elif 'state_dict' in loaded and isinstance(loaded['state_dict'], dict):
                state_dict = loaded['state_dict']
            else:
                # Filter out metadata keys
                state_dict = {k: v for k, v in loaded.items() if k != '_metadata'}
            
            model.load_state_dict(state_dict, strict=False)
        elif hasattr(loaded, 'state_dict'):
            model.load_state_dict(loaded.state_dict(), strict=False)
        else:
            model.load_state_dict(loaded, strict=False)
        
        return True
    except Exception as e:
        print(f"    Error loading state dict: {e}")
        return False


def load_config_from_dir(seed_dir: Path) -> Optional[Dict]:
    """Load config.json from seed directory."""
    config_path = seed_dir / 'config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return None


# =============================================================================
# Sample generation functions
# =============================================================================

def generate_ffm_samples(
    model: FNO,
    dataset_info: Dict,
    n_samples: int,
    device: str,
    config: Optional[Dict] = None,
) -> torch.Tensor:
    """Generate samples using FFM model."""
    # Get GP prior params from config or defaults
    kernel_length = dataset_info['kernel_length']
    kernel_variance = dataset_info['kernel_variance']
    
    if config:
        kernel_length = config.get('gp_kernel_length', kernel_length)
        kernel_variance = config.get('gp_kernel_variance', kernel_variance)
    
    # Create FFM wrapper
    ffm = FFMModelOT(
        model=model,
        kernel_length=kernel_length,
        kernel_variance=kernel_variance,
        sigma_min=1e-4,
        use_ot=False,  # Not needed for sampling
        device=device,
        dtype=torch.float32,
    )
    
    # Generate samples
    spatial_dims = dataset_info['spatial_dims']
    n_channels = dataset_info.get('n_channels', 1)
    
    if dataset_info['is_2d']:
        samples = ffm.sample(list(spatial_dims), n_samples=n_samples, n_channels=n_channels)
        samples = samples.cpu().squeeze(1)  # Remove channel dim if present
    else:
        samples = ffm.sample(list(spatial_dims), n_samples=n_samples)
        samples = samples.cpu().squeeze()
    
    return samples


def generate_ddpm_samples(
    model: FNO,
    dataset_info: Dict,
    n_samples: int,
    device: str,
) -> torch.Tensor:
    """Generate samples using DDPM model."""
    ddpm = DiffusionModel(
        model,
        method='DDPM',
        T=1000,
        device=device,
        kernel_length=dataset_info['kernel_length'],
        kernel_variance=dataset_info['kernel_variance'],
        beta_min=1e-4,
        beta_max=0.02,
    )
    
    spatial_dims = dataset_info['spatial_dims']
    n_channels = dataset_info.get('n_channels', 1)
    
    if dataset_info['is_2d']:
        samples = ddpm.sample(list(spatial_dims), n_samples=n_samples, n_channels=n_channels)
        samples = samples.cpu().squeeze(1)
    else:
        samples = ddpm.sample(list(spatial_dims), n_samples=n_samples)
        samples = samples.cpu().squeeze()
    
    return samples


def generate_ncsn_samples(
    model: FNO,
    dataset_info: Dict,
    n_samples: int,
    device: str,
) -> torch.Tensor:
    """Generate samples using NCSN/DDO model."""
    # NCSN hyperparameters
    sigma1 = 100.0 if dataset_info['is_2d'] else 1.0
    
    ncsn = DiffusionModel(
        model,
        method='NCSN',
        T=10,
        device=device,
        kernel_length=dataset_info['kernel_length'],
        kernel_variance=dataset_info['kernel_variance'],
        sigma1=sigma1,
        sigmaT=0.01,
        precondition=True,
    )
    
    spatial_dims = dataset_info['spatial_dims']
    n_channels = dataset_info.get('n_channels', 1)
    
    if dataset_info['is_2d']:
        samples = ncsn.sample(list(spatial_dims), n_samples=n_samples, n_channels=n_channels)
        samples = samples.cpu().squeeze(1)
    else:
        samples = ncsn.sample(list(spatial_dims), n_samples=n_samples)
        samples = samples.cpu().squeeze()
    
    return samples


# =============================================================================
# Main scanning and generation logic
# =============================================================================

def scan_for_models(outputs_dir: Path, dataset_key: Optional[str] = None) -> List[Dict]:
    """
    Scan output directories for trained models.
    
    Returns list of dicts with:
        - dataset: dataset key
        - config_name: config folder name
        - seed_dir: path to seed directory
        - model_path: path to model.pt
        - samples_path: path to samples.pt (may not exist)
        - has_samples: whether samples.pt exists
        - method_type: FFM, DDPM, NCSN, or GANO
    """
    results = []
    
    datasets_to_scan = [dataset_key] if dataset_key else list(DATASETS.keys())
    
    for ds_key in datasets_to_scan:
        if ds_key not in DATASETS:
            continue
        
        ds_info = DATASETS[ds_key]
        ds_dir = outputs_dir / ds_info['dir']
        
        if not ds_dir.exists():
            continue
        
        # Handle datasets with subdirectories (e.g., economy)
        subdirs = ds_info.get('subdirs', [None])
        
        for subdir in subdirs:
            search_dir = ds_dir / subdir if subdir else ds_dir
            if not search_dir.exists():
                continue
            
            # Scan for config directories
            for config_dir in search_dir.iterdir():
                if not config_dir.is_dir():
                    continue
                
                config_name = config_dir.name
                method_type = detect_method_type(config_name)
                
                # Scan for seed directories
                for seed_dir in config_dir.iterdir():
                    if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
                        continue
                    
                    model_path = seed_dir / 'model.pt'
                    samples_path = seed_dir / 'samples.pt'
                    
                    # Also check for GANO model files
                    gano_model_path = seed_dir / 'model_G.pt'
                    
                    if model_path.exists() or gano_model_path.exists():
                        results.append({
                            'dataset': ds_key,
                            'config_name': config_name,
                            'seed_dir': seed_dir,
                            'model_path': model_path if model_path.exists() else gano_model_path,
                            'samples_path': samples_path,
                            'has_samples': samples_path.exists(),
                            'method_type': method_type,
                            'subdir': subdir,
                        })
    
    return results


def generate_samples_for_model(
    model_info: Dict,
    n_samples: int,
    device: str,
    force: bool = False,
) -> bool:
    """
    Generate samples for a single model.
    
    Returns True if samples were generated successfully.
    """
    if model_info['has_samples'] and not force:
        return True  # Already has samples
    
    ds_key = model_info['dataset']
    ds_info = DATASETS[ds_key]
    method_type = model_info['method_type']
    seed_dir = model_info['seed_dir']
    model_path = model_info['model_path']
    
    print(f"  Generating samples for {model_info['config_name']}/{seed_dir.name}...")
    
    # Skip GANO for now (needs both G and D models)
    if method_type == 'GANO':
        print(f"    Skipping GANO model (requires special handling)")
        return False
    
    # Create model
    try:
        if ds_info['is_2d']:
            model = create_model_2d(
                ds_info['modes'],
                ds_info['hch'],
                ds_info['pch'],
                ds_info.get('t_scaling', 1000),
                device,
            )
        else:
            model = create_model_1d(
                ds_info['modes'],
                ds_info['width'],
                ds_info['mlp_width'],
                ds_info.get('t_scaling', 1000),
                device,
            )
        
        # Load state dict
        if not load_state_dict_flexible(model, model_path, device):
            return False
        
        model.eval()
        
        # Load config for GP params
        config = load_config_from_dir(seed_dir)
        
        # Generate samples based on method type
        with torch.no_grad():
            if method_type == 'DDPM':
                samples = generate_ddpm_samples(model, ds_info, n_samples, device)
            elif method_type == 'NCSN':
                samples = generate_ncsn_samples(model, ds_info, n_samples, device)
            else:  # FFM
                samples = generate_ffm_samples(model, ds_info, n_samples, device, config)
        
        # Save samples
        torch.save(samples, model_info['samples_path'])
        print(f"    ✓ Saved {samples.shape} samples to {model_info['samples_path']}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate samples from existing trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate missing samples for all datasets
    python generate_samples.py
    
    # Generate for specific dataset
    python generate_samples.py --dataset navier_stokes
    
    # Force regenerate even if samples exist
    python generate_samples.py --force
    
    # Dry run - just show what would be generated
    python generate_samples.py --dry-run
    
    # List available datasets
    python generate_samples.py --list-datasets
"""
    )
    parser.add_argument('--dataset', '-d', type=str, default=None,
                        help='Specific dataset to process (default: all)')
    parser.add_argument('--outputs-dir', '-o', type=str, default='../outputs',
                        help='Path to outputs directory')
    parser.add_argument('--n-samples', '-n', type=int, default=500,
                        help='Number of samples to generate (default: 500)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force regenerate even if samples exist')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just show what would be generated')
    parser.add_argument('--list-datasets', action='store_true',
                        help='List available datasets')
    
    args = parser.parse_args()
    
    if args.list_datasets:
        print("Available datasets:")
        for key, info in DATASETS.items():
            dims = 'x'.join(map(str, info['spatial_dims']))
            dim_type = '2D' if info['is_2d'] else '1D'
            print(f"  {key}: {info['dir']} ({dim_type}, {dims})")
        return
    
    # Setup
    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.exists():
        print(f"Outputs directory not found: {outputs_dir}")
        return
    
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Scan for models
    print("\n" + "=" * 60)
    print("Scanning for trained models...")
    print("=" * 60)
    
    models = scan_for_models(outputs_dir, args.dataset)
    
    if not models:
        print("No trained models found.")
        return
    
    # Group by dataset
    by_dataset = defaultdict(list)
    for m in models:
        by_dataset[m['dataset']].append(m)
    
    # Summary
    print(f"\nFound {len(models)} models across {len(by_dataset)} datasets:")
    for ds, ms in by_dataset.items():
        n_with_samples = sum(1 for m in ms if m['has_samples'])
        n_missing = len(ms) - n_with_samples
        print(f"  {ds}: {len(ms)} models ({n_missing} missing samples)")
    
    # Filter to models needing samples
    if args.force:
        to_generate = models
    else:
        to_generate = [m for m in models if not m['has_samples']]
    
    if not to_generate:
        print("\nAll models already have samples. Use --force to regenerate.")
        return
    
    print(f"\nWill generate samples for {len(to_generate)} models")
    
    if args.dry_run:
        print("\n[DRY RUN] Would generate samples for:")
        for m in to_generate:
            print(f"  - {m['dataset']}/{m['config_name']}/{m['seed_dir'].name} ({m['method_type']})")
        return
    
    # Generate samples
    print("\n" + "=" * 60)
    print("Generating samples...")
    print("=" * 60)
    
    success = 0
    failed = 0
    
    for m in to_generate:
        print(f"\n{m['dataset']}/{m['config_name']}")
        if generate_samples_for_model(m, args.n_samples, device, args.force):
            success += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"Done! Generated samples for {success} models, {failed} failed.")
    print("=" * 60)


if __name__ == '__main__':
    main()
