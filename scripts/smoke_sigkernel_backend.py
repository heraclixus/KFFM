#!/usr/bin/env python3
"""Smoke/benchmark signature-kernel backends on Heston-like paths."""

from __future__ import annotations

import argparse
import time

import torch


def make_paths(batch: int, length: int, dim: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    x = torch.randn(batch, length, dim, dtype=dtype, device=device)
    t = torch.linspace(0, 1, length, dtype=dtype, device=device).view(1, length, 1)
    t = t.expand(batch, length, 1)
    return torch.cat([t, x], dim=-1).contiguous()


def bench_sigkernel(x: torch.Tensor, y: torch.Tensor, dyadic_order: int, sigma: float, max_batch: int):
    import sigkernel

    static_kernel = sigkernel.RBFKernel(sigma=sigma)
    kernel = sigkernel.SigKernel(static_kernel, dyadic_order=dyadic_order)
    t0 = time.perf_counter()
    out = kernel.compute_Gram(x, y, sym=False, max_batch=max_batch)
    if out.device.type == "cuda":
        torch.cuda.synchronize(out.device)
    return time.perf_counter() - t0, out


def bench_pysiglib(x: torch.Tensor, y: torch.Tensor, dyadic_order: int, sigma: float, max_batch: int):
    from pysiglib.static_kernels import RBFKernel
    from pysiglib.torch_api import sig_kernel_gram

    static_kernel = RBFKernel(sigma)
    t0 = time.perf_counter()
    out = sig_kernel_gram(
        x,
        y,
        dyadic_order=dyadic_order,
        time_aug=False,
        static_kernel=static_kernel,
        max_batch=max_batch,
    )
    if out.device.type == "cuda":
        torch.cuda.synchronize(out.device)
    return time.perf_counter() - t0, out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--length", type=int, default=65)
    parser.add_argument("--dim", type=int, default=1)
    parser.add_argument("--dyadic-order", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable; using cpu")
        device = "cpu"

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}")
    print(
        f"batch={args.batch} length={args.length} dim={args.dim} "
        f"dyadic_order={args.dyadic_order} max_batch={args.max_batch} dtype={args.dtype}"
    )

    x = make_paths(args.batch, args.length, args.dim, dtype, device)
    y = make_paths(args.batch, args.length, args.dim, dtype, device)

    results = {}
    for name, fn in (("sigkernel", bench_sigkernel), ("pysiglib", bench_pysiglib)):
        try:
            dt, out = fn(x, y, args.dyadic_order, args.sigma, args.max_batch)
            print(
                f"{name}: {dt:.4f}s shape={tuple(out.shape)} dtype={out.dtype} "
                f"device={out.device} mean={float(out.float().mean()):.6g}"
            )
            results[name] = out.detach().cpu().float()
        except Exception as exc:
            print(f"{name}: ERROR {type(exc).__name__}: {exc}")

    if "sigkernel" in results and "pysiglib" in results:
        diff = (results["sigkernel"] - results["pysiglib"]).abs()
        print(f"abs_diff: mean={float(diff.mean()):.6g} max={float(diff.max()):.6g}")


if __name__ == "__main__":
    main()
