# kFFM: Kernel Functional Flow Matching

Official code for **Improving Function Space Flow Matching with Kernel Optimal Transport** (NeurIPS 2026).

Fred Xu, Thomas Markovich, Barbora Barancikova, Yizhou Sun

Functional Flow Matching (FFM) pairs each prior sample with an arbitrary data function. kFFM replaces this
independent pairing with an entropic optimal transport (OT) coupling under a kernel-induced cost. The
neural-operator architecture is unchanged; only the minibatch pairing of prior and data samples changes.

## Coupling costs and base measures

| Name in the paper | `ot_kernel` | Cost |
|---|---|---|
| kFFM-Sig | `signature` | signature kernel on time-augmented paths |
| kFFM-Sob | `sobolev_rbf` | RBF kernel on a Sobolev norm |
| kFFM-RBF | `rbf` | RBF kernel on Euclidean distances |
| kFFM-Euc | `euclidean` | raw L2 cost |

The base measure is a Gaussian-process prior (`use_gp_prior: true`) or white noise (`use_gp_prior: false`).
The finite-dimensional baseline CFM-OT(L2) is the raw L2 cost with the white-noise base.

By default the cost matrix is median-normalized per batch, Sinkhorn runs in the log domain, and endpoint
pairs are sampled from the plan. See `docs/optimal_transport.md` and `docs/hyperparams.md`.

## Installation

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`:

```bash
uv sync
```

`environment.yml` is the conda environment used for the signature-kernel experiments, which rely on
[pySigLib](https://github.com/daniil-shmelev/pySigLib).

## Project structure

```
├── functional_fm.py        # FFM (independent coupling)
├── functional_fm_ot.py     # kFFM: FFM with kernel OT coupling
├── optimal_transport.py    # kernels, cost matrices, Sinkhorn plans, plan sampling
├── diffusion.py            # DDPM and DDO/NCSN baselines
├── gano.py, gano1d.py      # GANO baseline
├── conditional_ffm.py      # conditional FFM
├── optimal_ffm.py          # optimal functional flow matching with convex potentials
├── models/                 # Fourier neural operators and baseline networks
├── util/                   # GP priors, data loading, evaluation metrics
├── configs/                # sweep and baseline configurations
├── scripts/                # training, evaluation, table and figure scripts
├── tests/                  # unit tests
└── docs/                   # notes on OT variants, hyperparameters, diagnostics
```

## Data

Datasets are not distributed with the code. The loaders expect the following files under `data/`:

| Dataset | File | Grid |
|---|---|---|
| AEMET | `aemet.csv` | 365 points |
| Gene expression | `full_genes.pt` | |
| Economy (population, GDP, labor) | `economy/econ1.pt`, `econ2.pt`, `econ3.pt` | |
| KdV | `KdV.mat` | 512 points, 201 snapshots of one trajectory |
| Stochastic KdV | `stochastic_kdv.mat` | 128 points |
| Navier-Stokes | `ns.mat` | 64 x 64 |
| Stochastic Navier-Stokes | `stochastic_ns_64.mat` | 64 x 64 |
| Turbulent Navier-Stokes (viscosity 1e-5) | `NavierStokes_V1e-5_N1200_T20.mat` | 64 x 64 |
| Navier-Stokes, cost profiling only | `nsforcing_128/` | 128 x 128 |
| Stochastic Ginzburg-Landau | `stochastic_ginzburg_landau.mat` | 129 points |

Heston and rBergomi paths are simulated (100 time steps, 1000 for the long variants) and cached under
`data/cache/`. The AEMET, gene expression and economy data come from the
[FFM repository](https://github.com/GavinKerrigan/functional_flow_matching), the PDE data from
[torchspde](https://github.com/crispitagorico/torchspde), and the turbulent Navier-Stokes data from the
Fourier neural operator benchmark.

## Quick start

Run all commands from `scripts/`.

```bash
# one dataset, all coupling configurations
python AEMET_ot.py
python navier_stokes_ot.py

# multi-seed runs of one configuration
python run_seeded_experiments.py --list
python run_seeded_experiments.py --dataset aemet --kernel signature --n_seeds 10

# non-kernel metrics, paired tests, tables
python compute_distributional_metrics.py
python audit_uniform20_mmd.py
python generate_tables.py --type sequence
python generate_tables.py --type pde
```

Other entry points: `profile_coupling.py` (time and memory of the coupling step),
`measure_delta_kappa.py` (kernel-cost mismatch), `toy_coupling_example.py` (synthetic couplings),
`plot_hyperparam_sensitivity.py` (sensitivity sweeps).

## Tests

```bash
pytest
```

## Citation

```bibtex
@inproceedings{xu2026kffm,
  title     = {Improving Function Space Flow Matching with Kernel Optimal Transport},
  author    = {Xu, Fred and Markovich, Thomas and Barancikova, Barbora and Sun, Yizhou},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgments and license

This code builds on the official implementation of
[Functional Flow Matching](https://github.com/GavinKerrigan/functional_flow_matching) by Kerrigan et al.
and on [POT](https://pythonot.github.io/) for optimal transport. Released under the MIT license; see
[LICENSE](LICENSE).
