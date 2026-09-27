"""Toy examples where L2 pairing fails and the kernel costs give the intended
matches.

Three scenarios:

1. **Shifted bumps** (easy control, full grid only): localized Gaussian
   bumps at random shifts; intended match = nearest shift. All costs recover
   the monotone matching -- kernel couplings do not break cases where L2 is
   already right.

2. **Bumps + heavy-tailed bursts** (L2 conditioning failure, Thm 1): same
   bumps, but 40% of samples on each side carry a large-amplitude nuisance
   burst at a random location (independent across sides, so bursts carry no
   matching signal). Burst pairs dominate the unbounded L2 cost's median
   normalization, so clean-pair contrasts collapse below eps and the L2 plan
   degenerates toward uniform. The bounded RBF cost saturates burst pairs
   (cost <= 2), the median stays at the clean scale, and the intended matching
   is recovered on the clean block -- invariant to burst amplitude.

3. **Volatility matching** (L2 geometry failure): f = v * BM(t) with
   per-sample volatility v ~ U[0.5, 2]; intended match = similar volatility
   (the generative factor, as in Heston). For independent Brownian
   paths E||f - g||^2 = (v^2 + w^2) * int t dt is *separable* in (v, w): every
   coupling has the same expected cost, so the L2 plan carries no assortative
   signal. The signature kernel measures v through its quadratic-variation
   levels, so its cost grows with |v - w| and the plan concentrates on the
   intended matching.

Diagnostic: plan mass on the intended-match band (|param_i - param'_j| small),
compared to the uniform-plan value.

--public: compact 2x2 figure -- row 1 = bursts (L2 vs Euclidean-RBF),
row 2 = volatility (L2 vs signature); per-panel uniform reference in titles.
Default: full grid (3 scenarios x 4 kernels), console references.

Run from scripts/:  conda run -n signature python toy_coupling_example.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimal_transport import KernelOTPlanSampler, create_kernel  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

B = 64
N_GRID = 128
EPS = 0.01
WIDTH = 0.03
BURST_FRAC = 0.4
BURST_AMP = 30.0

t = torch.linspace(0, 1, N_GRID)

KERNELS = {
    "l2": ("L2 cost (CFM-OT / kFFM-Euc)", lambda: None),
    "rbf": ("Euclidean-RBF kernel", lambda: create_kernel("rbf", sigma=1.0)),
    "sobolev": (
        "Sobolev-RBF kernel",
        lambda: create_kernel("sobolev_rbf", sigma=1.0, s=1.0),
    ),
    "sig": (
        "Signature kernel (lead-lag)",
        # Lead-lag exposes quadratic variation at signature level 2; the
        # linear static kernel reads it directly (RBF static dilutes the
        # QV signal with path-shape mismatch, ~0.36 vs ~0.40 band mass).
        lambda: create_kernel(
            "signature",
            time_aug=True,
            lead_lag=True,
            dyadic_order=1,
            static_kernel_type="linear",
            normalize=False,
            max_seq_len=N_GRID,
        ),
    ),
}


def bumps(taus):
    return torch.exp(-((t[None, :] - taus[:, None]) ** 2) / (2 * WIDTH**2))


# per-scenario generators: numbers are reproducible and independent of which
# scenarios run / in which order
def scenario_bumps(seed=0):
    g = torch.Generator().manual_seed(seed)
    tau0, _ = torch.sort(torch.rand(B, generator=g) * 0.7 + 0.15)
    tau1, _ = torch.sort(torch.rand(B, generator=g) * 0.7 + 0.15)
    x0 = bumps(tau0).unsqueeze(1)
    x1 = bumps(tau1).unsqueeze(1)
    band = (torch.abs(tau0[:, None] - tau1[None, :]) < 0.05).numpy()
    return "Shifted bumps (match by location)", x0, x1, band


def scenario_bursts(seed=0):
    g = torch.Generator().manual_seed(seed)
    tau0, _ = torch.sort(torch.rand(B, generator=g) * 0.7 + 0.15)
    tau1, _ = torch.sort(torch.rand(B, generator=g) * 0.7 + 0.15)

    def contaminate(x):
        n_b = int(BURST_FRAC * B)
        idx = torch.randperm(B, generator=g)[:n_b]
        centers = torch.rand(n_b, generator=g) * 0.7 + 0.15
        c = BURST_AMP * torch.exp(0.5 * torch.randn(n_b, generator=g))
        x[idx] = x[idx] + c[:, None] * torch.exp(
            -((t[None, :] - centers[:, None]) ** 2) / (2 * WIDTH**2)
        )
        return x

    x0 = contaminate(bumps(tau0)).unsqueeze(1)
    x1 = contaminate(bumps(tau1)).unsqueeze(1)
    band = (torch.abs(tau0[:, None] - tau1[None, :]) < 0.05).numpy()
    return "Bumps + heavy-tailed bursts (match by location)", x0, x1, band


def scenario_volatility(seed=0):
    g = torch.Generator().manual_seed(seed)
    v0, _ = torch.sort(torch.rand(B, generator=g) * 1.5 + 0.5)
    v1, _ = torch.sort(torch.rand(B, generator=g) * 1.5 + 0.5)

    def paths(vols):
        dW = torch.randn(B, N_GRID, generator=g) / np.sqrt(N_GRID)
        return vols[:, None] * torch.cumsum(dW, dim=1)

    x0 = paths(v0).unsqueeze(1)
    x1 = paths(v1).unsqueeze(1)
    band = (torch.abs(v0[:, None] - v1[None, :]) < 0.15).numpy()
    return "Brownian paths, v ~ U[0.5, 2] (match by volatility)", x0, x1, band


def main():
    public = "--public" in sys.argv
    sampler = KernelOTPlanSampler(
        method="sinkhorn", reg=EPS, normalize_cost=True, sinkhorn_max_iter=5000
    )
    if public:
        # each row features the kernel the scenario isolates
        rows = [
            (scenario_bursts(), ["l2", "rbf"]),
            (scenario_volatility(), ["l2", "sig"]),
        ]
    else:
        all_k = list(KERNELS)
        rows = [
            (scenario_bumps(), all_k),
            (scenario_bursts(), all_k),
            (scenario_volatility(), all_k),
        ]

    n_cols = max(len(ks) for _, ks in rows)
    fig, axes = plt.subplots(
        len(rows), n_cols, figsize=(4.5 * n_cols, 4 * len(rows)), squeeze=False
    )
    for row, ((sc_name, x0, x1, band), kkeys) in enumerate(rows):
        print(f"\n=== {sc_name} ===")
        uniform_mass = band.mean()
        print(f"uniform-plan band mass: {uniform_mass:.3f}")
        for col, kkey in enumerate(kkeys):
            kname, kfactory = KERNELS[kkey]
            pi = sampler.get_map(x0, x1, kernel_fn=kfactory())
            band_mass = float((pi * band).sum())
            print(f"{kname:<32} band mass = {band_mass:.3f}")
            ax = axes[row, col]
            ax.imshow(pi, cmap="viridis", aspect="auto")
            title = f"{kname}\nintended-match mass: {band_mass:.2f}"
            if public:
                title += f" (uniform: {uniform_mass:.2f})"
            ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{sc_name}\nsource (sorted)", fontsize=9)
            ax.set_xlabel("target (sorted)")
        for col in range(len(kkeys), n_cols):
            axes[row, col].axis("off")

    suffix = (
        "intended coupling = diagonal"
        if public
        else "uniform-plan reference in console output"
    )
    fig.suptitle(
        f"Entropic couplings (b={B}, eps={EPS}, median-normalized costs); {suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(__file__).parent.parent / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    name = "toy_coupling_compact.png" if public else "toy_coupling.png"
    fig.savefig(out / name, dpi=150)
    print(f"\nsaved {out / name}")


if __name__ == "__main__":
    main()
