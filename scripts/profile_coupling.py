"""G4: peak GPU memory + wall-clock of the kernel-OT coupling vs the FNO backbone.

For each benchmark shape and batch size, measures
  (a) FNO forward+backward (the per-step training cost that scales with the model), and
  (b) the coupling step: kernel Gram / cost matrix + Sinkhorn plan (KernelOTPlanSampler.get_map)
      for each kernel family used in the paper.
OOMs are recorded as results rather than crashing the sweep.

Run from scripts/:  python profile_coupling.py
Output: ../outputs/coupling_profile.json + PROFILE lines on stdout.
"""

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimal_transport import KernelOTPlanSampler, create_kernel  # noqa: E402
from run_seeded_experiments import (  # noqa: E402
    create_model_1d,
    create_model_2d,
    setup_heston,
    setup_navier_stokes,
    setup_navier_stokes_128,
    setup_stochastic_kdv,
)

DEVICE = "cuda"
BATCH_SIZES = [64, 128, 256, 512]
REPS = 3

BENCHMARKS = [
    # (name, setup_fn, grid shape, 1D signature kernel applicable)
    ("heston", setup_heston, (100,), True),
    ("stochastic_kdv", setup_stochastic_kdv, (512,), True),
    ("navier_stokes_64", setup_navier_stokes, (64, 64), False),
    ("navier_stokes_128", setup_navier_stokes_128, (128, 128), False),
]


def kernels_for(grid, with_sig):
    ks = [
        ("euclidean_L2", None),
        ("rbf", create_kernel("rbf", sigma=1.0)),
        ("sobolev_rbf", create_kernel("sobolev_rbf", sigma=1.0, s=1.0)),
    ]
    if with_sig:
        ks.append((
            "signature",
            create_kernel(
                "signature", time_aug=True, lead_lag=False, dyadic_order=1,
                static_kernel_sigma=1.0, max_seq_len=grid[0],
            ),
        ))
    return ks


def measure(fn):
    """Return (median wall ms, peak CUDA MB) over REPS after one warmup."""
    fn()
    torch.cuda.synchronize()
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(REPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return sorted(times)[len(times) // 2], peak_mb


def main():
    sampler = KernelOTPlanSampler(method="sinkhorn", reg=0.1, normalize_cost=True)
    results = []
    for name, setup_fn, grid, with_sig in BENCHMARKS:
        try:
            setup = setup_fn()
        except Exception as e:  # data missing etc. -- keep sweeping
            print(f"PROFILE bench={name} SETUP_FAILED {type(e).__name__}: {e}")
            continue
        if setup.get("is_2d", False):
            model = create_model_2d(setup["modes"], setup["hch"], setup["pch"], DEVICE)
        else:
            model = create_model_1d(setup["modes"], setup["width"], setup["mlp_width"], DEVICE)
        n_par = sum(p.numel() for p in model.parameters())
        for b in BATCH_SIZES:
            u = torch.randn(b, 1, *grid, device=DEVICE)
            t = torch.rand(b, device=DEVICE)

            def fwd_bwd():
                model.zero_grad(set_to_none=True)
                loss = model(t, u).pow(2).mean()
                loss.backward()

            try:
                ms, mb = measure(fwd_bwd)
                rec = dict(bench=name, b=b, item="fno_fwd_bwd", ms=ms, peak_mb=mb, n_params=n_par)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rec = dict(bench=name, b=b, item="fno_fwd_bwd", oom=True, n_params=n_par)
            results.append(rec)
            print("PROFILE", json.dumps(rec))

            x0 = torch.randn(b, 1, *grid, device=DEVICE)
            x1 = torch.randn(b, 1, *grid, device=DEVICE)
            for kname, kfn in kernels_for(grid, with_sig):
                def coupling():
                    sampler.get_map(x0, x1, kernel_fn=kfn)

                try:
                    ms, mb = measure(coupling)
                    rec = dict(bench=name, b=b, item=kname, ms=ms, peak_mb=mb)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    rec = dict(bench=name, b=b, item=kname, oom=True)
                except Exception as e:
                    rec = dict(bench=name, b=b, item=kname, error=f"{type(e).__name__}: {e}")
                results.append(rec)
                print("PROFILE", json.dumps(rec))
        del model
        torch.cuda.empty_cache()

    out = Path(__file__).parent.parent / "outputs" / "coupling_profile.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    print("PROFILE_DONE", out)


if __name__ == "__main__":
    main()
