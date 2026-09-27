"""
Heston Stochastic Volatility Model.

The Heston model is a classical stochastic volatility model where:
- dS_t = μ S_t dt + √V_t S_t dW^S_t
- dV_t = κ(θ - V_t) dt + σ √V_t dW^V_t
- Corr(dW^S, dW^V) = ρ

Parameters:
- μ: drift rate
- κ: mean reversion speed
- θ: long-term variance
- σ: vol-of-vol
- ρ: correlation (leverage effect)

Feller condition: 2κθ > σ² ensures V stays positive

Usage:
    python Heston.py                    # Generate default dataset
    python Heston.py --kappa 2.0        # Faster mean reversion
    python Heston.py --n_samples 10000  # Generate more samples
"""

import torch
import numpy as np
from pathlib import Path
import argparse


class HestonModel(torch.nn.Module):
    """
    Heston model class compatible with torchsde interface.
    """
    def __init__(self, mu, kappa, theta, sigma, rho):
        super(HestonModel, self).__init__()
        # Parameters as tensors
        self.mu = torch.tensor(mu, dtype=torch.float32)
        self.kappa = torch.tensor(kappa, dtype=torch.float32)
        self.theta = torch.tensor(theta, dtype=torch.float32)
        self.sigma = torch.tensor(sigma, dtype=torch.float32)
        self.rho = torch.tensor(rho, dtype=torch.float32)

        # Specify the noise type as 'general'
        self.noise_type = 'general'
        self.sde_type = 'ito'

    def f(self, t, y):
        # Drift part
        S, V = y[..., 0], y[..., 1]
        dS = self.mu * S
        dV = self.kappa * (self.theta - V)
        return torch.stack([dS, dV], dim=-1)

    def g(self, t, y):
        # Diffusion part corrected to account for noise dimensionality
        S, V = y[..., 0], y[..., 1]
        # Ensure V is non-negative for sqrt
        V_safe = torch.clamp(V, min=1e-8)
        vol_S = torch.sqrt(V_safe)
        vol_v = self.sigma * torch.sqrt(V_safe)

        # Constructing a tensor of shape (batch_size, state_dim, noise_dim)
        dW1_dS = vol_S * S  # dW1 effect on S
        dW1_dV = torch.zeros_like(S)  # dW1 has no direct effect on V

        dW2_dS = torch.zeros_like(S)  # dW2 has no direct effect on S
        dW2_dV = self.rho * vol_S + torch.sqrt(1 - self.rho ** 2) * vol_v  # dW2 effect on V

        # Stacking to get the correct shape: (batch, state_channels, noise_channels)
        return torch.stack([torch.stack([dW1_dS, dW1_dV], dim=-1),
                            torch.stack([dW2_dS, dW2_dV], dim=-1)], dim=-1)


def simulate_heston_euler(
    n_samples: int,
    n_steps: int,
    T: float,
    mu: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    S0: float = 1.0,
    V0: float = 0.04,
) -> dict:
    """
    Simulate Heston model paths using Euler-Maruyama scheme with reflection.
    
    Args:
        n_samples: Number of paths
        n_steps: Number of time steps
        T: Time horizon
        mu: Drift rate
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma: Vol-of-vol
        rho: Correlation
        S0: Initial price
        V0: Initial variance
    
    Returns:
        Dictionary with S, V, log_S, log_V paths
    """
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    
    # Initialize arrays
    S = np.zeros((n_samples, n_steps + 1))
    V = np.zeros((n_samples, n_steps + 1))
    
    S[:, 0] = S0
    V[:, 0] = V0
    
    # Correlation matrix for Brownian motions
    # dW^S and dW^V are correlated with correlation rho
    for i in range(n_steps):
        # Generate correlated Brownian increments
        dW1 = np.random.randn(n_samples) * sqrt_dt  # For S
        dZ = np.random.randn(n_samples) * sqrt_dt   # Independent
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dZ  # For V, correlated with dW1
        
        # Current variance (use reflection to keep positive)
        V_curr = np.maximum(V[:, i], 1e-8)
        sqrt_V = np.sqrt(V_curr)
        
        # Update price: dS = μ S dt + √V S dW^S
        S[:, i+1] = S[:, i] * np.exp(
            (mu - 0.5 * V_curr) * dt + sqrt_V * dW1
        )
        
        # Update variance: dV = κ(θ - V) dt + σ √V dW^V
        # Using reflection scheme for positivity
        V_new = V[:, i] + kappa * (theta - V_curr) * dt + sigma * sqrt_V * dW2
        V[:, i+1] = np.maximum(V_new, 1e-8)  # Reflection at 0
    
    # Convert to tensors
    S = torch.from_numpy(S).float()
    V = torch.from_numpy(V).float()
    t = torch.linspace(0, T, n_steps + 1)
    
    return {
        'S': S,
        'V': V,
        'log_S': torch.log(S),
        'log_V': torch.log(V),
        't': t,
    }


# Default parameters for standard and long sequences
HESTON_DEFAULT_STEPS = 100
HESTON_LONG_STEPS = 1000


def generate_heston_dataset(
    n_samples: int = 5000,
    n_steps: int = HESTON_DEFAULT_STEPS,
    T: float = 1.0,
    mu: float = 0.05,
    kappa: float = 1.0,
    theta: float = 0.04,
    sigma: float = 0.3,
    rho: float = -0.7,
    S0: float = 1.0,
    V0: float = 0.04,
    seed: int = 42,
    save_dir: Path = None,
) -> dict:
    """
    Generate Heston model dataset and optionally save to disk.
    
    Args:
        n_samples: Number of paths to generate
        n_steps: Number of time steps per path
        T: Time horizon
        mu: Drift rate
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma: Vol-of-vol
        rho: Correlation (leverage effect)
        S0: Initial price
        V0: Initial variance
        seed: Random seed
        save_dir: Directory to save dataset
    
    Returns:
        Dictionary containing generated paths and metadata
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Check Feller condition
    feller = 2 * kappa * theta / (sigma ** 2)
    print(f"Generating Heston dataset...")
    print(f"  Parameters: κ={kappa}, θ={theta}, σ={sigma}, ρ={rho}")
    print(f"  Feller condition 2κθ/σ² = {feller:.3f} {'> 1 ✓' if feller > 1 else '< 1 (V may hit 0)'}")
    print(f"  Samples: {n_samples}, Steps: {n_steps}, T: {T}")
    
    # Simulate paths
    paths = simulate_heston_euler(
        n_samples=n_samples,
        n_steps=n_steps,
        T=T,
        mu=mu,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        S0=S0,
        V0=V0,
    )
    
    V = paths['V']
    log_V = paths['log_V']
    S = paths['S']
    log_S = paths['log_S']
    t = paths['t']
    
    # Compute normalized versions for training
    log_V_normalized = (log_V - log_V.mean()) / log_V.std()
    log_S_normalized = (log_S - log_S.mean()) / log_S.std()
    
    print(f"  V shape: {V.shape}")
    print(f"  log_V range: [{log_V.min():.3f}, {log_V.max():.3f}]")
    print(f"  log_V_normalized std: {log_V_normalized.std():.3f}")
    
    dataset = {
        'V': V,
        'log_V': log_V,
        'log_V_normalized': log_V_normalized,
        'S': S,
        'log_S': log_S,
        'log_S_normalized': log_S_normalized,
        't': t,
        'params': {
            'mu': mu,
            'kappa': kappa,
            'theta': theta,
            'sigma': sigma,
            'rho': rho,
            'S0': S0,
            'V0': V0,
            'T': T,
            'n_steps': n_steps,
            'n_samples': n_samples,
            'seed': seed,
            'feller_ratio': feller,
        }
    }
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename based on parameters
        filename = f"Heston_kappa{kappa}_sigma{sigma}_n{n_samples}.pt"
        save_path = save_dir / filename
        
        torch.save(dataset, save_path)
        print(f"  Saved to: {save_path}")
    
    return dataset


def generate_heston_long_dataset(
    n_samples: int = 5000,
    n_steps: int = HESTON_LONG_STEPS,
    T: float = 1.0,
    mu: float = 0.05,
    kappa: float = 1.0,
    theta: float = 0.04,
    sigma: float = 0.3,
    rho: float = -0.7,
    S0: float = 1.0,
    V0: float = 0.04,
    seed: int = 42,
    save_dir: Path = None,
) -> dict:
    """
    Generate Heston model dataset with long time series (default 1000 steps).
    
    This is a convenience wrapper around generate_heston_dataset with
    n_steps set to 1000 by default for longer sequence modeling.
    
    Args:
        n_samples: Number of paths to generate
        n_steps: Number of time steps per path (default: 1000)
        T: Time horizon
        mu: Drift rate
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma: Vol-of-vol
        rho: Correlation (leverage effect)
        S0: Initial price
        V0: Initial variance
        seed: Random seed
        save_dir: Directory to save dataset
    
    Returns:
        Dictionary containing generated paths and metadata
    """
    return generate_heston_dataset(
        n_samples=n_samples,
        n_steps=n_steps,
        T=T,
        mu=mu,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        rho=rho,
        S0=S0,
        V0=V0,
        seed=seed,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Heston stochastic volatility dataset")
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of paths')
    parser.add_argument('--n_steps', type=int, default=None, help='Time steps per path (default: 100, or 1000 with --long)')
    parser.add_argument('--T', type=float, default=1.0, help='Time horizon')
    parser.add_argument('--mu', type=float, default=0.05, help='Drift rate')
    parser.add_argument('--kappa', type=float, default=1.0, help='Mean reversion speed')
    parser.add_argument('--theta', type=float, default=0.04, help='Long-term variance')
    parser.add_argument('--sigma', type=float, default=0.3, help='Vol-of-vol')
    parser.add_argument('--rho', type=float, default=-0.7, help='Correlation')
    parser.add_argument('--S0', type=float, default=1.0, help='Initial price')
    parser.add_argument('--V0', type=float, default=0.04, help='Initial variance')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='.', help='Save directory')
    parser.add_argument('--generate_all', action='store_true',
                        help='Generate datasets for multiple parameter combinations')
    parser.add_argument('--long', action='store_true',
                        help=f'Generate long time series (default: {HESTON_LONG_STEPS} steps)')
    
    args = parser.parse_args()
    
    # Determine n_steps: explicit > --long > default
    if args.n_steps is not None:
        n_steps = args.n_steps
    elif args.long:
        n_steps = HESTON_LONG_STEPS
    else:
        n_steps = HESTON_DEFAULT_STEPS
    
    save_dir = Path(args.save_dir)
    
    if args.generate_all:
        # Generate datasets with different parameter combinations
        configs = [
            # Varying mean reversion speed
            {'kappa': 0.5, 'sigma': 0.3, 'name': 'slow_reversion'},
            {'kappa': 1.0, 'sigma': 0.3, 'name': 'medium_reversion'},
            {'kappa': 2.0, 'sigma': 0.3, 'name': 'fast_reversion'},
            # Varying vol-of-vol
            {'kappa': 1.0, 'sigma': 0.2, 'name': 'low_volvol'},
            {'kappa': 1.0, 'sigma': 0.5, 'name': 'high_volvol'},
        ]
        
        variant = "long" if args.long else "standard"
        print("="*60)
        print(f"Generating Heston datasets ({variant}, {n_steps} steps)")
        print("="*60)
        
        for config in configs:
            print(f"\n--- {config['name']} ---")
            generate_heston_dataset(
                n_samples=args.n_samples,
                n_steps=n_steps,
                T=args.T,
                mu=args.mu,
                kappa=config['kappa'],
                theta=args.theta,
                sigma=config['sigma'],
                rho=args.rho,
                S0=args.S0,
                V0=args.V0,
                seed=args.seed,
                save_dir=save_dir,
            )
        
        print("\n" + "="*60)
        print("All datasets generated!")
        print("="*60)
    else:
        # Generate single dataset
        variant = "heston-long" if args.long else "heston"
        print(f"Generating {variant} dataset ({n_steps} steps)...")
        generate_heston_dataset(
            n_samples=args.n_samples,
            n_steps=n_steps,
            T=args.T,
            mu=args.mu,
            kappa=args.kappa,
            theta=args.theta,
            sigma=args.sigma,
            rho=args.rho,
            S0=args.S0,
            V0=args.V0,
            seed=args.seed,
            save_dir=save_dir,
        )
