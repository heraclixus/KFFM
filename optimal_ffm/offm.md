# Optimal Functional Flow Matching (OFM in Function Spaces) — Design & Formulation for Spatio-Temporal PDE Generation

This document records the mathematical formulation and design choices for an **Optimal Functional Flow Matching** model that generates **spatio-temporal PDE trajectories** on **periodic** domains (KdV 1D, NS 2D), focusing exclusively on **Hilbert-space** formulations (ignore Banach/path-space variants).

We aim to combine:
- **Functional Flow Matching (FFM)** style modeling in infinite-dimensional spaces, with
- **Optimal Flow Matching (OFM)** style restriction to **quadratic OT structure** using **convex potentials**.

---

## 0. Problem Setup

### Data
We observe trajectories (solutions) of PDEs on periodic spatial domains:
- **KdV (1D):** domain $D = [0, L)$
- **Navier–Stokes (2D):** domain $D = [0, L_x)\times[0, L_y)$

Trajectories are sampled on **uniform time** and **uniform space** grids:

- Time: $t_n = n\Delta t$, $n=0,\dots,N_t-1$, $T = (N_t-1)\Delta t$
- Space:
  - 1D: $x_i = i\Delta x$, $i=0,\dots,N_x-1$, $\Delta x = L/N_x$
  - 2D: $(x_i,y_j)$ with $\Delta x=L_x/N_x$, $\Delta y=L_y/N_y$

Each sample is a spatio-temporal field:
- 1D: $u \in \mathbb{R}^{N_t\times N_x}$
- 2D: $u \in \mathbb{R}^{N_t\times N_x\times N_y}$ (possibly with channels, e.g. vorticity or velocity components)

### Generative goal
Learn a generative model producing samples with distribution $\mu_1$ over trajectories $u(t,\cdot)$, using a **Gaussian reference** $\mu_0$ on a Hilbert space and an OFM-style OT-structured flow.

---

## 1. Hilbert Space Choice for Spatio-Temporal Trajectories

We model each *entire trajectory* as an element of a Hilbert space over time with Sobolev regularity in space.

### 1.1 Default trajectory space (Hilbert)
$$
\mathcal{H}_{\text{traj}} := L^2\!\big(0,T;\,H^s(D)\big)
$$
with norm
$$
\|u\|^2_{L^2_t H^s_x}
:= \int_0^T \|u(t,\cdot)\|^2_{H^s(D)}\,dt.
$$

**Discrete approximation (uniform time grid):**
$$
\|u\|^2_{L^2_t H^s_x}
\approx \Delta t \sum_{n=0}^{N_t-1}\|u_n\|^2_{H^s(D)},
\quad u_n(\cdot)=u(t_n,\cdot).
$$

### 1.2 Time-regularity enhanced trajectory space (Hilbert)
To encourage smoothness in time and suppress frame-to-frame jitter:
$$
\mathcal{H}_{\text{traj}} :=
H^1\!\big(0,T;\,H^{s-1}(D)\big)\ \cap\ L^2\!\big(0,T;\,H^s(D)\big).
$$

A standard Hilbert norm is:
$$
\|u\|^2_{\text{traj}} :=
\int_0^T \|u(t)\|^2_{H^s}\,dt
+\int_0^T \|\partial_t u(t)\|^2_{H^{s-1}}\,dt.
$$

**Discrete approximation:**
$$
\|u\|^2_{\text{traj}}
\approx
\Delta t\sum_{n=0}^{N_t-1}\|u_n\|^2_{H^s}
+
\Delta t\sum_{n=0}^{N_t-2}\left\|\frac{u_{n+1}-u_n}{\Delta t}\right\|^2_{H^{s-1}}.
$$

### 1.3 Channel structure
If trajectories have $C$ channels (e.g. velocity components), use the product Hilbert space:
$$
\mathcal{H}_{\text{traj}}^{(C)} := \underbrace{\mathcal{H}_{\text{traj}}\times\cdots\times\mathcal{H}_{\text{traj}}}_{C\ \text{times}},
\quad
\|u\|^2=\sum_{c=1}^C \|u^{(c)}\|^2.
$$

---

## 2. Spatial Sobolev Norm on Periodic Grids (FFT-based)

Because KdV and NS datasets are periodic, we use spectral definitions of Sobolev norms.

### 2.1 Fourier frequencies
Let FFT frequencies be:
- 1D: $k\in\{0,1,\dots,N_x/2,-N_x/2+1,\dots,-1\}$ scaled by $2\pi/L$
- 2D: $(k_x,k_y)$ similarly scaled by $2\pi/L_x, 2\pi/L_y$

### 2.2 Discrete Sobolev norms
Let $\widehat{u}_n$ denote FFT coefficients of snapshot $u_n$.

**1D:**
$$
\|u_n\|^2_{H^s(D)}
\approx
\Delta x \sum_{k} (1+|k|^2)^s\,|\widehat{u}_n(k)|^2.
$$

**2D:**
$$
\|u_n\|^2_{H^s(D)}
\approx
\Delta x\Delta y \sum_{k_x,k_y} (1+|k|^2)^s\,|\widehat{u}_n(k_x,k_y)|^2,
\quad |k|^2=k_x^2+k_y^2.
$$

> Notes:
> - Using $(1+|k|^2)^s$ avoids issues at $k=0$.
> - Setting $s=0$ recovers an $L^2$ norm (Parseval).
> - For NS, one may use vorticity or divergence-free velocity representation; the above is channel-agnostic.

---

## 3. Reference Gaussian Measure on the Hilbert Trajectory Space

We choose a **Gaussian prior** $\mu_0 = \mathcal{N}(0,\mathcal{C})$ on $\mathcal{H}_{\text{traj}}$.

### 3.1 Practical parameterizations of $\mathcal{C}$ (choices)

#### Choice A: Factorized in time and space (most common)
$$
\mathcal{C} \approx \mathcal{C}_t \otimes \mathcal{C}_x.
$$
Discrete implementation: sample white noise $\xi$ on the grid and apply separable filters in time and space.

#### Choice B: Spectral diagonal covariance (periodic-friendly)
In Fourier domain (space, and optionally time), set:
$$
\mathbb{E}|\widehat{u}(k)|^2 \propto (1+|k|^2)^{-\alpha}
$$
so samples have desired smoothness. This is convenient for Sobolev control.

#### Choice C: Simple isotropic Gaussian on the discrete Hilbert space
Treat trajectory as a vector in $\mathbb{R}^{N_t N_x}$ or $\mathbb{R}^{N_t N_x N_y}$, use $\mu_0 = \mathcal{N}(0,\sigma^2 I)$, and rely on the model + Sobolev cost to impose structure.
This is simplest but least physically informed.

---

## 4. Quadratic OT Geometry on $\mathcal{H}_{\text{traj}}$

We consider the quadratic cost induced by the trajectory Hilbert norm:
$$
c(u,v) := \frac12 \|u-v\|^2_{\mathcal{H}_{\text{traj}}}.
$$

The **optimal transport** problem between $\mu_0$ and $\mu_1$ is:
$$
\inf_{\pi\in\Pi(\mu_0,\mu_1)}\ \int \frac12\|u-v\|^2_{\mathcal{H}_{\text{traj}}}\,d\pi(u,v).
$$

When a Monge map $T$ exists, the OT displacement interpolation is:
$$
U_t = (1-t)U_0 + t\,T(U_0),\quad U_0\sim\mu_0,\quad \mu_t = (U_t)_\#\mu_0.
$$

---

## 5. Optimal Functional Flow Matching Objective (OFM-style)

We parameterize a **convex functional** $\Phi:\mathcal{H}_{\text{traj}}\to\mathbb{R}$ and learn it via an OFM-style regression through an inversion (proximal) operator.

### 5.1 OFM inversion / proximal map on Hilbert space
Define, for $t\in(0,1]$, the **prox-like inverse**:
$$
\widehat{U}_0(U_t,t;\Phi)
:=
\arg\min_{z\in\mathcal{H}_{\text{traj}}}
\left\{
\frac12\|z\|^2_{\mathcal{H}_{\text{traj}}}
-\langle U_t,z\rangle_{\mathcal{H}_{\text{traj}}}
+ t\,\Phi(z)
\right\}.
$$

If $\Phi$ is convex and lower-semicontinuous, the objective is strongly convex due to the $\tfrac12\|z\|^2$ term, so the minimizer is unique.

### 5.2 Training distribution for interpolation points
Given a coupling $\pi$ between $\mu_0$ and $\mu_1$, sample:
- $(U_0, U_1)\sim \pi$
- $t \sim \mathrm{Unif}[0,1]$
- $U_t = (1-t)U_0 + tU_1$

### 5.3 OFM loss (functional regression)
$$
\mathcal{L}(\Phi)
=
\mathbb{E}_{(U_0,U_1)\sim\pi,\ t\sim\mathrm{Unif}}
\left[
\left\|
\widehat{U}_0(U_t,t;\Phi)-U_0
\right\|^2_{\mathcal{H}_{\text{traj}}}
\right].
$$

**Intuition:** learn a convex potential so that the inverse map recovers the reference sample $U_0$ from an interpolated point $U_t$. Under OT structure, $\Phi$ corresponds to a dual potential (quadratic OT).

---

## 6. Parameterizing a Convex Functional $\Phi$ (design choices)

We require $\Phi$ convex in the Hilbert argument $u\in\mathcal{H}_{\text{traj}}$. Below are convex-by-construction options.

### 6.1 Basis projection + ICNN (recommended baseline)
Choose a bounded linear map $P:\mathcal{H}_{\text{traj}}\to \mathbb{R}^m$ (e.g. truncated Fourier coefficients across space and time, or PCA modes).
Define:
$$
\Phi(u) = \varphi(Pu) + \frac{\lambda}{2}\|u\|^2_{\mathcal{H}_{\text{traj}}},
$$
where $\varphi:\mathbb{R}^m\to\mathbb{R}$ is an **Input-Convex Neural Network (ICNN)** and $\lambda\ge 0$.

- Convexity holds because $P$ is linear and ICNN is convex.
- $\lambda>0$ yields strong convexity and stabilizes the inversion.

**Choices for $P$:**
- Space FFT truncation per time (keep low spatial frequencies)
- Space-time FFT truncation (keep low frequencies in both)
- POD/PCA modes from data (linear, compact)
- Multi-resolution wavelets (linear)

### 6.2 Convex neural operator pattern (Fourier features + pointwise convex integrand)
Incorporate operator-like linear spectral features (FNO-style *linear* convolutions) while preserving convexity:
$$
\Phi(u) =
\sum_{n,i} w\, \psi\Big( (K_1u)_{n,i},\dots,(K_ru)_{n,i},\ u_{n,i},\ \text{coords}_{n,i}\Big)
+ \frac{\lambda}{2}\|u\|^2_{\mathcal{H}_{\text{traj}}}.
$$
- The $K_j$ are **linear** spectral operators (Fourier multipliers / filters / derivatives).
- $\psi$ is **convex** in its numeric arguments (ICNN).
- Summation preserves convexity.
- Works for 1D and 2D; indices become $(n,i)$ or $(n,i,j)$.

This yields “operator awareness” without breaking convexity.

### 6.3 Quadratic + convex residual (fast & stable)
Let $\Phi$ be a quadratic energy plus a convex learned correction:
$$
\Phi(u) = \frac12\langle u,Au\rangle + \varphi(Pu),
$$
with $A\succeq 0$ (e.g. Sobolev operator in Fourier domain). This makes the inversion step easier and can be solved efficiently (often with conjugate gradients).

---

## 7. Solving the Inversion Subproblem

The inversion defines:
$$
z^\star = \arg\min_z\ \frac12\|z\|^2 - \langle U_t,z\rangle + t\Phi(z).
$$

### 7.1 If $\Phi(u)=\varphi(Pu)+\frac{\lambda}{2}\|u\|^2$
The objective becomes:
$$
\frac{1+\lambda t}{2}\|z\|^2 - \langle U_t,z\rangle + t\varphi(Pz).
$$
This is strongly convex. Solve with:
- gradient-based methods (L-BFGS / Adam) if dimensions are moderate
- proximal / splitting methods if $\varphi$ has structure
- implicit differentiation through the optimizer if needed

### 7.2 If $\Phi$ is pointwise-sum of convex $\psi$
The gradient is local (plus linear operators), still convex. Use:
- accelerated gradient / L-BFGS
- warm-start using previous iterations (for efficiency)

> Implementation tip: since training samples are many, the inversion must be efficient. Favor architectures where the inverse solve is cheap (quadratic + convex residual; low-dimensional $P$; or few iterations with warm starts).

---

## 8. Generating Samples

Once $\Phi$ is learned, we need a forward sampling procedure to map $U_0\sim\mu_0$ to $U_1\sim\mu_1$.

### 8.1 Explicit OT map approximation (conceptual)
In quadratic OT on a Hilbert space, the Monge map often has the form:
$$
T(u) = u - \nabla \Phi(u)
$$
(up to conventions / dual potentials). In practice, we may approximate a map $T_\theta$ induced by $\Phi$ via the learned proximal/inverse relation.

### 8.2 Practical sampling options

#### Option S1: Learn a direct transport map from $\Phi$
Given $\Phi$, define a map:
$$
T_\Phi(u) := \operatorname{prox}_{\Phi}(u) \quad \text{or} \quad u-\nabla\Phi(u)
$$
depending on how $\Phi$ is defined in the objective and how the OFM inversion corresponds to the OT map.

#### Option S2: Use the learned “inverse” as a building block
Because training learns $\widehat U_0(U_t,t)$, one can construct a consistent forward map by solving the corresponding forward optimality condition.

#### Option S3: Use an ODE in the Hilbert space
Define a time-dependent velocity field consistent with the OT displacement:
$$
\frac{d}{dt}U_t = T(U_0)-U_0,
$$
and sample along straight lines once $T$ is available. (In practice this reduces to straight interpolation in $\mathcal{H}_{\text{traj}}$.)

> The exact forward sampling recipe depends on the precise OFM convention adopted (dual potential vs primal). Record and keep consistent with the final derivation.

---

## 9. Possible Choices Summary (Checklist)

### Trajectory Hilbert space
- [ ] $L^2(0,T;H^s(D))$ (default)
- [ ] $H^1(0,T;H^{s-1}(D))\cap L^2(0,T;H^s(D))$ (time-smooth trajectories)
- [ ] Multi-channel product space

### Sobolev norm evaluation (periodic)
- [ ] FFT-based spectral Sobolev $H^s$
- [ ] Select $s$ (e.g., $0,1,2$) depending on desired smoothness/physics

### Gaussian reference $\mu_0$
- [ ] Spectral diagonal covariance enforcing smoothness
- [ ] Separable time⊗space covariance
- [ ] Simple isotropic Gaussian on discretized vector (baseline)

### Convex functional $\Phi$
- [ ] Projection $P$ + ICNN + optional quadratic regularizer
- [ ] Convex neural operator: linear Fourier multipliers + pointwise ICNN + sum/integral
- [ ] Quadratic operator energy + convex learned residual

### Inversion solver
- [ ] L-BFGS / accelerated gradient
- [ ] Warm-start
- [ ] Reduce dimension via low-rank $P$

### Training interpolation coupling $\pi$
- [ ] Independent coupling: $U_0\sim\mu_0$, $U_1\sim\mu_1$ independent
- [ ] More informed coupling (optional): e.g., match low-frequencies, class-conditional, etc.

---



## 10. Notes Specific to KdV (1D) and NS (2D)

### KdV (1D periodic)
- Use 1D FFT for $H^s$ norms.
- Often scalar field $u(t,x)$.

### NS (2D periodic)
- Use 2D FFT.
- State representation choices:
  - vorticity scalar field $\omega(t,x,y)$
  - velocity $(u,v)$ with divergence-free constraint (can be enforced via streamfunction or projection in Fourier domain)

If using divergence-free velocity, incorporate the projection operator (linear in Fourier domain) inside the linear feature maps $K_j$ or inside the Hilbert norm definition.

---

## 11. Minimal Mathematical Statement (for paper)

**Hilbert trajectory space:** $\mathcal{H}_{\text{traj}} = L^2(0,T;H^s(D))$ with periodic $D$.  
**Cost:** $c(u,v)=\tfrac12\|u-v\|^2_{\mathcal{H}_{\text{traj}}}$.  
**Reference:** $\mu_0=\mathcal{N}(0,\mathcal{C})$ on $\mathcal{H}_{\text{traj}}$.  
**Target:** $\mu_1$ is the empirical distribution of PDE trajectories.  
**Convex potential:** $\Phi:\mathcal{H}_{\text{traj}}\to\mathbb{R}$ parameterized convex-by-construction.  
**Inverse map:**
$$
\widehat{U}_0(U_t,t;\Phi)
=
\arg\min_{z\in\mathcal{H}_{\text{traj}}}
\left\{\frac12\|z\|^2-\langle U_t,z\rangle+t\Phi(z)\right\}.
$$
**Loss:**
$$
\mathcal{L}(\Phi)
=
\mathbb{E}\left[\|\widehat{U}_0(U_t,t;\Phi)-U_0\|^2\right],
\quad U_t=(1-t)U_0+tU_1.
$$

---

## 12. Open Implementation Questions (to resolve later)

1. Exact convention linking $\Phi$ to the OT map $T$ in the functional setting (dual vs primal potential).
2. Most efficient inversion solver and number of iterations per minibatch.
3. Best low-dimensional linear projection $P$ balancing expressiveness and compute (space FFT truncation vs space-time vs POD).
4. For NS velocity representation, whether to generate vorticity or divergence-free velocity directly.

---

## Appendix A: Discrete Inner Product Definition (explicit)

Define the discrete trajectory inner product approximating $L^2_t H^s_x$:
$$
\langle u,v\rangle_{\text{traj}}
:=
\Delta t \sum_{n=0}^{N_t-1} \langle u_n,v_n\rangle_{H^s(D)},
$$
with
- 1D:
  $$
  \langle u_n,v_n\rangle_{H^s}
  :=
  \Delta x \sum_k (1+|k|^2)^s\,\widehat{u}_n(k)\overline{\widehat{v}_n(k)}.
  $$
- 2D:
  $$
  \langle u_n,v_n\rangle_{H^s}
  :=
  \Delta x\Delta y \sum_{k_x,k_y}(1+|k|^2)^s\,\widehat{u}_n(k_x,k_y)\overline{\widehat{v}_n(k_x,k_y)}.
  $$

Optional time-derivative term:
$$
\langle u,v\rangle_{H^1_t H^{s-1}_x}
:=
\Delta t\sum_{n=0}^{N_t-2}
\left\langle \frac{u_{n+1}-u_n}{\Delta t},\ \frac{v_{n+1}-v_n}{\Delta t}\right\rangle_{H^{s-1}}.
$$




# Practical implementation: OFM loss with stop-gradient (Sobolev trajectory Hilbert space)

This section mirrors the “stop-grad” trick used in practical OFM implementations, but adapted to our **functional / spatio-temporal** setting where each sample is a **trajectory** $u(t,\cdot)$ on a periodic domain (KdV 1D, NS 2D), represented on **uniform space–time grids**.

### Why stop-grad?
The OFM objective involves an **inner convex optimization** (an argmin / proximal map). Differentiating through the optimizer is:
- memory-heavy (unrolling many solver steps),
- numerically fragile (second-order info),
- unnecessary if we use an envelope/KKT-based objective.

Stop-gradient gives a stable, cheap training loop:
1) solve the inner convex problem to get $z^\star$,
2) **detach** $z^\star$,
3) update $\theta$ using a loss whose gradient does **not** require $\frac{\partial z^\star}{\partial \theta}$.

---

## 1 Discrete Hilbert structure (what “quadratic OT” uses)

We work in the discrete trajectory Hilbert space approximating $L^2(0,T;H^s(D))$ (and optionally an $H^1_t$ term).

For a trajectory $u$ (1D or 2D), define:

- **Trajectory norm (default)**:
  $$
  \|u\|^2_{\text{traj}} \approx \Delta t\sum_{n=0}^{N_t-1} \|u_n\|^2_{H^s(D)}.
  $$

- **Optional time-smoothness**:
  $$
  \|u\|^2_{\text{traj}} \approx
  \Delta t\sum_n \|u_n\|^2_{H^s}
  + \beta\,\Delta t\sum_{n=0}^{N_t-2}\left\|\frac{u_{n+1}-u_n}{\Delta t}\right\|^2_{H^{s-1}}.
  $$

- **Periodic Sobolev norm via FFT** (for each snapshot $u_n$):
  - 1D:
    $$
    \|u_n\|^2_{H^s} \approx \Delta x \sum_k (1+|k|^2)^s |\widehat u_n(k)|^2.
    $$
  - 2D:
    $$
    \|u_n\|^2_{H^s} \approx \Delta x\Delta y \sum_{k_x,k_y} (1+|k|^2)^s |\widehat u_n(k_x,k_y)|^2.
    $$

All inner products $\langle\cdot,\cdot\rangle_{\text{traj}}$ are induced by these norms.

---

## 2 Inner convex problem (prox / inverse step)

Given an interpolated point $U_t$ and time $t\in(0,1]$, define the inner objective
$$
g_\theta(z;U_t,t)
:= \frac12\|z\|^2_{\text{traj}} - \langle U_t, z\rangle_{\text{traj}} + t\,\Phi_\theta(z),
$$
and compute the unique minimizer
$$
z^\star = \arg\min_{z} g_\theta(z;U_t,t).
$$

**First-order optimality (KKT):**
$$
0 = \nabla_z g_\theta(z^\star;U_t,t)
= z^\star - U_t + t\,\nabla \Phi_\theta(z^\star),
$$
so equivalently:
$$
U_t = z^\star + t\,\nabla\Phi_\theta(z^\star).
$$

This identity is extremely useful for stop-grad training.

---

## 3) Two practical stop-grad losses (recommended)

### Loss A: “Energy gap” loss (envelope-friendly)
We want the true $U_0$ (from the reference) to be the minimizer of $g_\theta(\cdot;U_t,t)$ under the correct potential.

Define:
$$
\mathcal{L}_{\text{gap}}(\theta)
:= \mathbb{E}\big[g_\theta(U_0;U_t,t) - g_\theta(z^\star;U_t,t)\big].
$$

Because $z^\star$ is the minimizer, the gap is always $\ge 0$, and pushing it to $0$ enforces that $U_0$ attains the minimum.

**Stop-grad rule:** compute $z^\star$ with the solver, then **detach** it in the loss:
- gradients flow through $\Phi_\theta(U_0)$ and $\Phi_\theta(z^\star)$,
- we do **not** backprop through the solver steps.

This matches an “envelope theorem” intuition: for a minimized objective, $\frac{d}{d\theta}\min_z g_\theta(z)$ can be taken as $\partial_\theta g_\theta(z^\star)$ without $\partial_\theta z^\star$.

---

### Loss B: “KKT residual” loss (cheap and very stable)
Use the optimality condition as a regression constraint:
$$
\mathcal{L}_{\text{KKT}}(\theta)
:= \mathbb{E}\Big[\|U_t - z^\star - t\,\nabla\Phi_\theta(z^\star)\|^2_{\text{traj}}\Big].
$$

**Stop-grad rule:** detach $z^\star$, but still compute $\nabla\Phi_\theta(z^\star)$ (gradient wrt input) and backprop wrt $\theta$.

This loss often stabilizes training early and can be combined with the gap loss:
$$
\mathcal{L}(\theta) = \mathcal{L}_{\text{gap}}(\theta) + \gamma\,\mathcal{L}_{\text{KKT}}(\theta).
$$

> In practice:  
> - **Gap loss** enforces the “correct minimizer” property (ties to OFM derivation).  
> - **KKT loss** enforces the stationarity relation and reduces solver error sensitivity.

---

## 4 Training-time sampling (spatio-temporal trajectories)

For each minibatch:
1. Sample reference trajectories $U_0 \sim \mu_0$ (Gaussian on the discrete trajectory Hilbert space).
2. Sample data trajectories $U_1 \sim \mu_1$ (empirical batch).
3. Sample $t \sim \mathrm{Unif}(\varepsilon, 1)$ (avoid $t=0$; e.g. $\varepsilon=10^{-3}$).
4. Form straight interpolation in the Hilbert space:
   $$
   U_t = (1-t)U_0 + tU_1.
   $$
5. Solve $z^\star = \arg\min_z g_\theta(z;U_t,t)$ (inner convex solve).
6. Compute stop-grad losses (detach $z^\star$ in the backward pass).

**Coupling choice:** simplest is independent coupling (sample $U_0$ and $U_1$ independently). More informed couplings are optional, but not required for the method description.

---

## 5) Inner solver (how we compute $z^\star$)

### Gradient-based convex solve (generic, easy)
Minimize $g_\theta(z;U_t,t)$ via a few steps of GD / accelerated GD / L-BFGS.

You need:
- the gradient $\nabla_z g_\theta(z;U_t,t) = z - U_t + t\nabla\Phi_\theta(z)$,
- the Hilbert geometry is implemented by using the discrete inner products/norms (FFT-weighted $H^s$ + time quadrature).

**Initialization:**
- good default: $z^{(0)} = U_t$,
- or warm-start from the previous solve if you reuse the same $t$ / nearby samples.

**Fixed-iteration solve (recommended for batching):**
Use $K$ steps (e.g., 5–20) with a tuned step size. Since we stop-grad, we don’t care about exact solver gradients, but we do care about $z^\star$ being reasonably close to the minimizer.

### Stop-grad implementation detail
During the inner solve, **do not build a huge autograd graph**. Typical pattern:
- compute $\nabla\Phi_\theta(z)$ for the update,
- update $z \leftarrow z - \eta (z-U_t+t\nabla\Phi_\theta(z))$,
- **detach** $z$ after each update (or run the solve under `torch.no_grad()` and only keep the final $z^\star$).

You only need autograd for:
- $\Phi_\theta(U_0)$,
- $\Phi_\theta(z^\star)$,
- $\nabla\Phi_\theta(z^\star)$ (for KKT loss),

not for the solve trajectory.

---

## 6 Pseudocode (stop-grad OFM functional training)

```python
# Inputs:
#   Phi_theta: convex functional network (ICNN / convex operator functional)
#   sample_mu0(batch): samples Gaussian reference trajectories U0
#   sample_mu1(batch): samples data trajectories U1
#   inner_solve(Ut, t, Phi_theta): returns approx minimizer z_star of g_theta(z;Ut,t)
# Hyperparams: eps_t, gamma, K (inner steps), step_size, etc.

for batch in loader:
    U0 = sample_mu0(B)          # [B, Nt, Nx] or [B, Nt, Nx, Ny, C]
    U1 = sample_mu1(B)

    t = Uniform(eps_t, 1.0).sample([B, 1, 1, ...])  # broadcastable
    Ut = (1 - t) * U0 + t * U1

    z_star = inner_solve(Ut, t, Phi_theta)          # convex solve
    z_det  = stopgrad(z_star)                       # detach for backward

    # g_theta(z;Ut,t) = 0.5||z||^2 - <Ut,z> + t Phi_theta(z)
    g_U0   = 0.5 * norm_traj(U0)**2 - inner_traj(Ut, U0) + t * Phi_theta(U0)
    g_z    = 0.5 * norm_traj(z_det)**2 - inner_traj(Ut, z_det) + t * Phi_theta(z_det)

    L_gap  = mean(g_U0 - g_z)

    # Optional KKT residual loss
    gradPhi_z = grad_wrt_input(Phi_theta, z_det)    # ∇Phi_theta(z_det)
    resid = Ut - z_det - t * gradPhi_z
    L_kkt = mean(norm_traj(resid)**2)

    loss = L_gap + gamma * L_kkt
    loss.backward()
    opt.step()
    opt.zero_grad()
```