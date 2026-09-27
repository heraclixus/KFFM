# Velocity Field Analysis for Flow Matching Models

This document describes the velocity field analysis metrics used to evaluate sampling stability in flow matching models.

## Overview

In flow matching, a neural network learns a velocity field $v_\theta(t, x)$ that transports samples from a base distribution (Gaussian Process prior) to the data distribution. The ODE:

$$\frac{dx}{dt} = v_\theta(t, x_t), \quad x_0 \sim \mu_0$$

is integrated from $t=0$ to $t=1$ to generate samples.

The **velocity field norm** $\|v_\theta(t, x_t)\|$ during this integration reveals important properties about sampling stability and efficiency.

## Metrics

### 1. Flow Uniformity (Velocity Uniformity)

**Definition**: The variance of mean velocity norms across time.

**Computation**:

1. Sample $N$ trajectories from the GP prior: $\{x_0^{(i)}\}_{i=1}^N$

2. For each time point $t_j \in \{0, \delta, 2\delta, \ldots, 1\}$:
   - Integrate each trajectory to time $t_j$: $x_{t_j}^{(i)} = \text{ODESolve}(x_0^{(i)}, t_j)$
   - Compute velocity norm: $\|v_\theta(t_j, x_{t_j}^{(i)})\|_2$
   - Compute mean over samples: $\bar{v}(t_j) = \frac{1}{N}\sum_{i=1}^N \|v_\theta(t_j, x_{t_j}^{(i)})\|$

3. Compute flow uniformity as the variance over time:
   $$\text{Uniformity} = \text{Var}_{t}[\bar{v}(t)] = \frac{1}{T}\sum_{j=1}^T (\bar{v}(t_j) - \bar{\bar{v}})^2$$
   
   where $\bar{\bar{v}} = \frac{1}{T}\sum_j \bar{v}(t_j)$ is the grand mean.

**Interpretation**:

| Value | Meaning |
|-------|---------|
| **Low uniformity** | Velocity magnitude is roughly constant over time → straighter paths |
| **High uniformity** | Velocity varies significantly over time → curved/non-uniform paths |

**Why it matters**:
- Straighter paths allow the ODE solver to take larger time steps
- This leads to fewer function evaluations (NFE) and faster sampling
- In finite dimensions OT coupling is often motivated by more uniform flows; the paper makes no such claim in function space, so this metric is an exploratory diagnostic

### 2. Number of Function Evaluations (NFE)

**Definition**: The number of times the neural network is called during ODE integration.

**Computation**:

The `dopri5` (Dormand-Prince 5(4)) adaptive ODE solver adjusts step sizes to meet specified tolerances. NFE is simply counted during integration:

```python
class VelocityTracker(nn.Module):
    def __init__(self, model):
        self.model = model
        self.nfe = 0
    
    def forward(self, t, x):
        self.nfe += 1
        return self.model(t, x)
```

**Interpretation**:
- Lower NFE = faster sampling
- NFE depends on:
  - ODE solver tolerances (lower tolerance → more NFE)
  - Smoothness/straightness of the learned flow
  - Data dimensionality

**Reference values** (from FFM paper):
- 1D datasets with `rtol=atol=1e-10`: ~600-700 NFE
- 2D datasets with `rtol=atol=1e-5`: ~60-150 NFE

### 3. Convergence Rate

**Definition**: The average rate of log-loss decrease during the first half of training.

**Computation**:

Given training losses $\{L_1, L_2, \ldots, L_E\}$ over $E$ epochs:

$$\text{Convergence Rate} = \frac{\log(L_1) - \log(L_{E/2})}{E/2}$$

This measures how many log-units the loss decreases per epoch on average during the initial training phase.

**Interpretation**:

| Value | Meaning |
|-------|---------|
| **Higher rate** | Loss decreases faster → faster convergence (better) |
| **Lower rate** | Loss decreases slowly → slower convergence |

**Note**: Unlike NFE and Uniformity, **higher convergence rate is better**.

**Why it matters**:
- Faster convergence means less training time needed
- OT coupling may help the model find better gradients by matching samples more appropriately
- A well-matched coupling can lead to more consistent gradient signals

### 4. Relationship Between Metrics

```
Flow Uniformity ↓  ⟹  Straighter paths  ⟹  Larger ODE steps  ⟹  NFE ↓
Convergence Rate ↑  ⟹  Faster training  ⟹  Better gradient signal
```

If k-FFM (kernel flow matching with OT coupling) produces:
- **Lower flow uniformity** than independent FFM
- **Lower NFE** than independent FFM
- **Higher convergence rate** than independent FFM

then the coupling is associated with more efficient transport on that dataset. This is an empirical diagnostic: kernel OT changes only the endpoint coupling, and the paper makes no straightness or sampling-efficiency claim.

## Implementation Details

### Scripts

- **`scripts/analyze_velocity_field.py`**: Computes metrics from trained model checkpoints
- **`scripts/plot_velocity_uniformity.py`**: Generates comparison plots

### Usage

```bash
# Analyze a single dataset (10 runs for mean ± std)
python analyze_velocity_field.py \
    --scan-dir ../outputs/seeded_runs/aemet \
    --dataset aemet \
    --n-runs 10

# Generate comparison plots
python plot_velocity_uniformity.py \
    --datasets aemet rbergomi heston kdv stochastic_kdv
```

### Output

Results are saved to `outputs/velocity_analysis/{dataset}/`:
- `{dataset}_velocity_analysis.json`: Raw metrics
- `{dataset}_velocity_trajectory.pdf`: Velocity norms over time
- `{dataset}_velocity_distribution.pdf`: Velocity norm distributions at time slices
- `{dataset}_stability_summary.pdf`: Summary bar charts

## Mathematical Background

### Conditional Flow Matching

The model learns to approximate the conditional velocity field:

$$u_t(x | x_1) = \frac{x_1 - (1 - \sigma_{\min}) x_t}{1 - (1 - \sigma_{\min}) t}$$

where $x_t = t \cdot x_1 + (1 - (1 - \sigma_{\min}) t) \cdot x_0$ is the interpolated sample.

### Finite-dimensional intuition

The following is the usual motivation for OT coupling in finite dimensions. It does not transfer automatically to function space, where kFFM keeps the affine FFM path and changes only the pairing of endpoints.

With **independent coupling** $(x_0, x_1) \sim \mu_0 \times \mu_1$:
- Training pairs may be far apart
- Learned flows can be curved to handle mismatched pairs

With **OT coupling** $(x_0, x_1) \sim \pi^*$ where $\pi^*$ minimizes transport cost:
- Training pairs are optimally matched
- Flows can be more direct
- Lower variance in velocity magnitudes over time

## References

1. Kerrigan et al., "Functional Flow Matching" (FFM)
2. Lipman et al., "Flow Matching for Generative Modeling"
3. Tong et al., "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport" (OT-CFM)
