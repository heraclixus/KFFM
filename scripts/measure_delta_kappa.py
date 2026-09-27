"""Empirical kernel-cost mismatch (Delta_kappa) measurement.

Measures when the kernel-cost mismatch of the error decomposition is small
(the appendix section "Measuring the Kernel-Cost Mismatch" of the paper).

For each dataset it draws training-faithful batches -- x0 from the same prior
used in training (GP prior, or iid Gaussian for the CFM arms) and x1 from the
training data, at the training batch size (capped for CPU) -- then compares the
raw L2 cost matrix c with the RKHS cost c_kappa of the bounded RBF kernel at
several bandwidths, plus the signature kernel on sequence datasets. Reported
per (dataset, kernel):

  spearman   rank correlation between the b^2 entries of c and c_kappa
             (the entropic plan depends on costs only through soft rankings,
             so high spearman => nearly identical couplings)
  rel_dev    ||c_hat_kappa - c_hat||_F / ||c_hat||_F after median
             normalization (exactly the normalization used in training)
  plan_tv    total-variation distance between the entropic plans
             (sinkhorn, reg=0.1 on the normalized costs, as in training)
  row_agree  fraction of rows whose argmax partner coincides across plans

Run from the repo's scripts/ directory (data paths are relative):
  conda run -n signature python measure_delta_kappa.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimal_transport import KernelOTPlanSampler, create_kernel  # noqa: E402
from util.util import make_grid  # noqa: E402
from util.gaussian_process import GPPrior  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

import os

REPEATS = 3
MAX_BATCH = 256
SIGMAS = (0.5, 1.0, 2.0)
SINKHORN_REG = float(os.environ.get("DK_REG", "0.1"))
ONLY = [d for d in os.environ.get("DK_DATASETS", "").split(",") if d]

# (setup function name, training-tuned sigma if known, try signature kernel)
DATASETS = {
    "kdv": ("setup_kdv", 1.0, False),
    "stochastic_kdv": ("setup_stochastic_kdv", 0.5, False),
    "heston": ("setup_heston", 2.0, True),
    "aemet": ("setup_aemet", 1.0, True),
    "expr_genes": ("setup_expr_genes", 1.0, True),
    "navier_stokes": ("setup_navier_stokes", 1.0, False),
    "navier_stokes_v1e5": ("setup_navier_stokes_v1e5", 1.0, False),
}


def get_setup(name):
    import run_seeded_experiments as rse

    return getattr(rse, name)()


def sample_prior(setup, batch, device="cpu"):
    """GP-prior batch matching training (all Table-1 kFFM rows use the GP prior)."""
    dims = (
        tuple(setup["spatial_dims"]) if setup.get("is_2d") else (setup["n_x"],)
    )
    gp = GPPrior(
        lengthscale=setup["kernel_length"],
        var=setup["kernel_variance"],
        device=device,
    )
    grid = make_grid(dims)
    z = gp.sample(grid, dims, n_samples=batch, n_channels=1)
    return z


def compare_costs(sampler, x0, x1, kernel_fn):
    """Return (spearman, rel_dev, plan_tv, row_agree) for c vs c_kappa."""
    c = sampler.compute_euclidean_cost(x0, x1)
    ck = sampler.compute_kernel_cost(x0, x1, kernel_fn)

    rho = stats.spearmanr(
        c.flatten().numpy(), ck.flatten().numpy()
    ).correlation

    c_hat = c / c.median()
    ck_hat = ck / ck.median()
    rel_dev = (torch.norm(ck_hat - c_hat) / torch.norm(c_hat)).item()

    pi_l2 = sampler.get_map(x0, x1, kernel_fn=None)
    pi_k = sampler.get_map(x0, x1, kernel_fn=kernel_fn)
    plan_tv = 0.5 * np.abs(pi_l2 - pi_k).sum()
    row_agree = float(
        (pi_l2.argmax(axis=1) == pi_k.argmax(axis=1)).mean()
    )
    return rho, rel_dev, plan_tv, row_agree


def main():
    sampler = KernelOTPlanSampler(
        method="sinkhorn", reg=SINKHORN_REG, normalize_cost=True
    )
    print(f"sinkhorn reg = {SINKHORN_REG}")
    results = {}

    for ds, (setup_name, tuned_sigma, try_sig) in DATASETS.items():
        if ONLY and ds not in ONLY:
            continue
        print(f"\n=== {ds} ===", flush=True)
        try:
            setup = get_setup(setup_name)
        except Exception as e:
            print(f"  SKIP (setup failed: {e})")
            continue

        data = setup["train_data"]
        batch = min(setup["batch_size"], MAX_BATCH, data.shape[0])
        results[ds] = {"batch": batch, "kernels": {}}

        kernels = [(f"rbf_s{s}", create_kernel("rbf", sigma=s)) for s in SIGMAS]
        if tuned_sigma not in SIGMAS:
            kernels.append(
                (f"rbf_s{tuned_sigma}(tuned)", create_kernel("rbf", sigma=tuned_sigma))
            )
        if try_sig:
            try:
                kernels.append(
                    (
                        "signature",
                        create_kernel(
                            "signature",
                            time_aug=True,
                            dyadic_order=1,
                            static_kernel_sigma=1.0,
                            max_seq_len=64,
                        ),
                    )
                )
            except Exception as e:
                print(f"  signature kernel unavailable: {e}")

        # typical distance scale, to contextualize sigma
        idx = torch.randperm(data.shape[0])[:batch]
        x1 = data[idx].float()
        x0 = sample_prior(setup, batch)
        d_med = sampler.compute_euclidean_cost(x0, x1).median().sqrt().item()
        results[ds]["median_l2_dist"] = d_med
        print(f"  batch={batch}  median ||x0-x1|| = {d_med:.3f}")

        for kname, kfn in kernels:
            rows = []
            for r in range(REPEATS):
                idx = torch.randperm(data.shape[0])[:batch]
                x1 = data[idx].float()
                x0 = sample_prior(setup, batch)
                try:
                    rows.append(compare_costs(sampler, x0, x1, kfn))
                except Exception as e:
                    print(f"  {kname}: FAILED ({e})")
                    rows = []
                    break
            if not rows:
                continue
            arr = np.array(rows)
            m = arr.mean(axis=0)
            results[ds]["kernels"][kname] = {
                "spearman": m[0],
                "rel_dev": m[1],
                "plan_tv": m[2],
                "row_agree": m[3],
            }
            print(
                f"  {kname:<20} spearman={m[0]:.4f}  rel_dev={m[1]:.3f}  "
                f"plan_tv={m[2]:.3f}  row_agree={m[3]:.2f}",
                flush=True,
            )

    suffix = "" if SINKHORN_REG == 0.1 else f"_reg{SINKHORN_REG}"
    out = (
        Path(__file__).parent.parent
        / "outputs"
        / f"delta_kappa_measurement{suffix}.json"
    )
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
