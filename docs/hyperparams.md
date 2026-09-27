# Hyperparameter Search Grids

This document summarizes the hyperparameter search ranges used in the sensitivity analysis plots (`scripts/plot_hyperparam_sensitivity.py`). Only the **top 10 configurations per kernel** (ranked by mean MSE) are displayed in the violin plots.

## Kernel Types

Four kernel types are compared in the sensitivity analysis:

- **None (Independent)**: No optimal transport pairing; baseline FFM
- **Euclidean**: L2 distance in function space for OT cost
- **RBF**: Radial Basis Function kernel for OT cost
- **Signature**: Path signature kernel for sequential/temporal data

---

## Common OT Hyperparameters

### Sinkhorn Regularization (`ot_reg` / ε)
- **Search range**: `{0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0}`
- **Description**: Entropic regularization for Sinkhorn OT algorithm
- **Trade-off**: Lower = more precise OT coupling but slower convergence; Higher = faster but more blurry transport

### OT Method (`ot_method`)
- **Options**: `{exact, sinkhorn, gaussian}`
- **`exact`**: Linear programming solver (no regularization, slower)
- **`sinkhorn`**: Entropic regularization (fast, default)
- **`gaussian`**: Bures-Wasserstein for Gaussian distributions (closed-form, very fast)

### Coupling Method (`ot_coupling`)
- **Options**: `{sample, barycentric}`
- **`sample`**: Sample-based coupling (default)
- **`barycentric`**: Barycentric projection coupling

---

## RBF Kernel Hyperparameters

### Bandwidth (`sigma`)
- **Search range**: `{0.5, 1.0, 2.0, 5.0, 10.0}`
- **Description**: RBF kernel bandwidth parameter controlling similarity scale
- **Interpretation**: Smaller σ = more local similarity; Larger σ = more global similarity

---

## Signature Kernel Hyperparameters

### Dyadic Order (`dyadic_order`)
- **Search range**: `{0, 1, 2, 3}`
- **Description**: Truncation level for the PDE approximation of signature kernel
- **Trade-off**: Higher order = captures more complex path interactions but is more computationally expensive
- **Dataset-specific**:
  - Smooth data (AEMET): `0-2` typically sufficient
  - Rough paths: lower orders `0-1` often work better

### Lead-Lag Augmentation (`lead_lag`)
- **Options**: `{true, false}`
- **Description**: Augments paths with lead-lag transformation to capture lagged dependencies
- **Recommended**: `true` for data with strong autocorrelation or rough paths

### Static Kernel Sigma (`static_kernel_sigma`)
- **Search range**: `{0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 5.0, 10.0}`
- **Description**: Bandwidth for the static kernel used in signature kernel computation
- **Dataset-specific**:
  - Rough paths: Smaller values `0.1-0.5` capture fine-scale roughness
  - Smooth seasonal data: Larger values `2.0-10.0` capture global patterns

### Static Kernel Type (`static_kernel_type`)
- **Options**: `{rbf, linear}`
- **Description**: Base kernel type for computing signature kernel

### Max Sequence Length (`max_seq_len`)
- **Search range**: `{32, 50, 64, 100, 128, 256}`
- **Description**: Maximum sequence length for signature computation (subsampling)
- **Memory-accuracy trade-off**: Longer = more accurate but more memory

### Time Augmentation (`time_aug`)
- **Options**: `{true, false}`
- **Description**: Whether to augment path with time coordinate
- **Default**: `true` (recommended for most cases)

### Normalize (`normalize`)
- **Options**: `{true, false}`
- **Description**: Whether to normalize the signature kernel
- **Default**: `true`

### Add Basepoint (`add_basepoint`)
- **Options**: `{true, false}`
- **Description**: Whether to add a basepoint at the origin
- **Default**: `true`

### Max Batch (`max_batch`)
- **Search range**: `{8, 16, 32, 64}`
- **Description**: Batch size for signature kernel computation (GPU memory constraint)

---

## Dataset-Specific Configurations

### Sequence Datasets (AEMET, Economy, Heston, Expr Genes)

| Hyperparameter | Typical Range |
|----------------|---------------|
| `ot_reg` | 0.01 - 1.0 |
| `dyadic_order` | 0 - 2 |
| `static_kernel_sigma` | 0.1 - 5.0 |
| `max_seq_len` | 64 - 128 |

### PDE Datasets (KdV, Navier-Stokes, Stochastic KdV, Stochastic NS)

| Hyperparameter | Typical Range |
|----------------|---------------|
| `ot_reg` | 0.1 - 1.0 |
| RBF `sigma` | 1.0 - 5.0 |

**Note**: Signature kernel is generally **not used** for PDE datasets due to:
- 2D spatial structure (Navier-Stokes)
- High computational cost for long sequences
- PDE solutions often lack the sequential structure signatures excel at

---

## Best Configurations per Dataset Category

### Stochastic Volatility Paths (Heston)
```yaml
# Top performer for volatility paths
kernel: signature
dyadic_order: 0-1
lead_lag: true
static_kernel_sigma: 0.2-0.5
ot_reg: 0.01-0.05
```

### Seasonal Time Series (AEMET)
```yaml
# Top performer for smooth seasonal data
kernel: signature
dyadic_order: 2
lead_lag: true/false
static_kernel_sigma: 2.0-5.0
ot_reg: 0.05-0.1
```

### 2D PDEs (Navier-Stokes, Stochastic NS)
```yaml
# Top performer for 2D spatial fields
kernel: euclidean or rbf
ot_reg: 0.1-0.5
ot_method: sinkhorn
```

---

## Filtering in Sensitivity Plots

The sensitivity plots in `scripts/plot_hyperparam_sensitivity.py` apply the following filtering:

1. **Top-K filtering**: Only the top 10 best configurations per kernel (sorted by `mean_mse`) are included
2. **Baseline exclusion**: DDPM and NCSN baselines are excluded from kernel comparisons
3. **Gaussian OT exclusion**: Gaussian OT method is excluded from kernel-specific analysis (different mathematical framework)
4. **Architecture variations excluded**: FNO architecture hyperparameters (modes, width) are not included in kernel sensitivity analysis

---

## References

- Signature kernel: [Kiraly & Oberhauser, 2019](https://arxiv.org/abs/1905.08494)
- Sinkhorn OT: [Cuturi, 2013](https://arxiv.org/abs/1306.0895)
- Bures-Wasserstein (Gaussian OT): [Bhatia et al., 2019](https://arxiv.org/abs/1705.02399)
